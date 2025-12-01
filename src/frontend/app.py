import streamlit as st
import pandas as pd
import numpy as np
import time
import os

# Configuración de la página
st.set_page_config(page_title="Fantasy Fantasy Simulator", layout="wide")

# ==========================================
# 1. GESTIÓN DE DATOS Y ESTADO
# ==========================================

@st.cache_data
def load_data():
    """Carga y prepara los datos desde el CSV"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Ajusta esta ruta si tu archivo está en otro lugar
    CSV_PATH = os.path.join(BASE_DIR, "../data/df.csv")
    
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ No se encontró el archivo CSV en: {CSV_PATH}")
        # Retorna un dataframe vacío para evitar crashes
        return pd.DataFrame(columns=['Player', 'Position', 'FPTS', 'FPTS/G'])

    df = pd.read_csv(CSV_PATH)
    
    # Verificación de seguridad
    # Necesitamos Player, Position, FPTS y FPTS/G
    required_cols = ['Player', 'Position', 'FPTS', 'FPTS/G']
    missing = [col for col in required_cols if col not in df.columns]
    
    if missing:
        st.error(f"❌ Faltan columnas en tu CSV: {missing}. Verifica los nombres exactos.")
        return pd.DataFrame()

    # Rellenar nulos
    df['FPTS'] = df['FPTS'].fillna(0)
    df['FPTS/G'] = df['FPTS/G'].fillna(0)
    
    return df.sort_values('FPTS', ascending=False).reset_index(drop=True)

# Cargar datos
df = load_data()

if df.empty:
    st.warning("No se pudieron cargar datos. Revisa la ruta del archivo CSV.")
    st.stop()

# Inicialización de Session State
if 'draft_state' not in st.session_state:
    st.session_state.draft_state = {
        'started': False,
        'finished': False,
        'my_team': [],
        'drafted_players': set(), # Set de nombres para búsqueda rápida
        'draft_log': [], # Lista de dicts {pick_num, team, player, position, points}
        'current_pick_overall': 1,
        'user_pick_slot': 1,
        'total_teams': 12,
        'roster_size': 14  # 9 titulares + 5 banca (Total 14)
    }

# ==========================================
# 2. LÓGICA DE NEGOCIO (HELPERS)
# ==========================================

def get_pick_owner(pick_num, total_teams, user_slot):
    """Calcula de quién es el pick en formato Snake Draft"""
    round_num = (pick_num - 1) // total_teams
    pick_in_round = (pick_num - 1) % total_teams + 1
    
    # Lógica Snake
    if round_num % 2 == 0:
        # Rounds 1, 3, 5... (Orden normal: 1 -> 12)
        current_picker = pick_in_round
    else:
        # Rounds 2, 4, 6... (Orden inverso: 12 -> 1)
        current_picker = total_teams - pick_in_round + 1
        
    is_user = (current_picker == user_slot)
    return current_picker, is_user, round_num + 1

def simulate_cpu_picks(df_available):
    """Simula picks hasta que sea el turno del usuario o termine el draft"""
    state = st.session_state.draft_state
    
    while not state['finished']:
        # Verificar si hay jugadores disponibles
        if df_available.empty:
            state['finished'] = True
            break

        current_owner, is_user, current_round = get_pick_owner(
            state['current_pick_overall'], state['total_teams'], state['user_pick_slot']
        )
        
        # Si es turno del usuario, paramos la simulación
        if is_user:
            break
            
        # Si el draft excedió el tamaño de roster * equipos
        if state['current_pick_overall'] > (state['total_teams'] * state['roster_size']):
            state['finished'] = True
            break
            
        # LÓGICA CPU: Pickear el mejor FPTS disponible
        best_player = df_available.iloc[0]
        
        # Registrar pick
        pick_data = {
            'pick': state['current_pick_overall'],
            'round': current_round,
            'team': f"Team {current_owner}",
            'player': best_player['Player'],
            'pos': best_player['Position'],
            'fpts': best_player['FPTS'],
            'fpts_g': best_player['FPTS/G']
        }
        
        state['draft_log'].append(pick_data)
        state['drafted_players'].add(best_player['Player'])
        state['current_pick_overall'] += 1
        
        # Actualizar dataframe local para el loop
        df_available = df_available.iloc[1:] 

def optimize_lineup(team_list):
    """
    Algoritmo de optimización usando FPTS.
    Requisitos: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DEF. Resto Banca.
    """
    if not team_list:
        return [], []

    df_team = pd.DataFrame(team_list)
    lineup = {'QB': [], 'RB': [], 'WR': [], 'TE': [], 'FLEX': [], 'K': [], 'DST': [], 'BENCH': []}
    
    # Ordenar por puntos totales (FPTS)
    df_team = df_team.sort_values('FPTS', ascending=False)
    
    # Buckets auxiliares
    rbs = df_team[df_team['Position'] == 'RB'].to_dict('records')
    wrs = df_team[df_team['Position'] == 'WR'].to_dict('records')
    tes = df_team[df_team['Position'] == 'TE'].to_dict('records')
    qbs = df_team[df_team['Position'] == 'QB'].to_dict('records')
    ks = df_team[df_team['Position'] == 'K'].to_dict('records')
    defs = df_team[df_team['Position'] == 'DST'].to_dict('records')

    # 1. Llenar titulares obligatorios
    if qbs: lineup['QB'].append(qbs.pop(0))
    
    while len(lineup['RB']) < 2 and rbs: lineup['RB'].append(rbs.pop(0))
    while len(lineup['WR']) < 2 and wrs: lineup['WR'].append(wrs.pop(0))
    if tes: lineup['TE'].append(tes.pop(0))
    if ks: lineup['K'].append(ks.pop(0))
    if defs: lineup['DST'].append(defs.pop(0))
    
    # 2. Calcular Flex (Mejor restante de RB, WR, TE)
    flex_pool = rbs + wrs + tes
    flex_pool = sorted(flex_pool, key=lambda x: x['FPTS'], reverse=True)
    
    if flex_pool:
        lineup['FLEX'].append(flex_pool.pop(0))
        
    # 3. Mover el resto a la banca
    starters_names = [p['Player'] for cat in lineup.values() for p in cat]
    
    for p in team_list:
        if p['Player'] not in starters_names:
            lineup['BENCH'].append(p)
            
    # Formatear para display
    starters = []
    order = ['QB', 'RB', 'WR', 'TE', 'FLEX', 'K', 'DST']
    for pos in order:
        starters.extend(lineup[pos])
        
    return starters, lineup['BENCH']

def calculate_power_rankings(draft_log, total_teams):
    """Calcula el total de puntos por equipo sin porcentajes"""
    if not draft_log:
        return pd.DataFrame()
    
    df_log = pd.DataFrame(draft_log)
    
    # Agrupar y sumar puntos
    team_stats = df_log.groupby('team')['fpts'].sum().reset_index()
    
    # Ordenar de mayor a menor
    team_stats = team_stats.sort_values('fpts', ascending=False).reset_index(drop=True)
    return team_stats

# ==========================================
# 3. INTERFAZ DE USUARIO
# ==========================================

st.title("🧙🏻‍♂️ Fantasy Football Simulator & Optimizer")

# Tabs para navegación
tab1, tab2, tab3, tab4 = st.tabs(["🏈 Draft Room", "📋 My Lineup & Optimizer", "⚖️ Trade Analyzer", "🏆 Power Rankings"])

# --- TAB 1: DRAFT ROOM ---
with tab1:
    st.subheader("Mock Draft Simulation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        draft_type = st.selectbox("Draft Type", ["Snake", "Auction (Coming Soon)"])
    with col2:
        num_teams = st.number_input("Number of Teams", 2, 32, 12)
    with col3:
        my_pick = st.number_input("Your Pick Position", 1, num_teams, 1)

    # Botón Reset
    if st.button("Start / Reset Draft"):
        st.session_state.draft_state['started'] = True
        st.session_state.draft_state['finished'] = False
        st.session_state.draft_state['my_team'] = []
        st.session_state.draft_state['drafted_players'] = set()
        st.session_state.draft_state['draft_log'] = []
        st.session_state.draft_state['current_pick_overall'] = 1
        st.session_state.draft_state['user_pick_slot'] = my_pick
        st.session_state.draft_state['total_teams'] = num_teams
        # Se actualiza el tamaño del roster
        st.session_state.draft_state['roster_size'] = 13 # Asegurar que tome el nuevo valor al reiniciar
        st.rerun()

    # Lógica del Draft en Vivo
    if st.session_state.draft_state['started'] and not st.session_state.draft_state['finished']:
        
        # Filtrar disponibles
        available_df = df[~df['Player'].isin(st.session_state.draft_state['drafted_players'])]
        
        # Simular picks de CPU antes de mi turno
        simulate_cpu_picks(available_df)
        
        # Recalcular disponibles después de simulación CPU
        available_df = df[~df['Player'].isin(st.session_state.draft_state['drafted_players'])]
        
        current_owner, is_user, current_round = get_pick_owner(
            st.session_state.draft_state['current_pick_overall'],
            st.session_state.draft_state['total_teams'],
            st.session_state.draft_state['user_pick_slot']
        )
        
        # Mostrar info del Pick Actual
        st.info(f"📍 Current Pick: {st.session_state.draft_state['current_pick_overall']} (Round {current_round}) - Owner: {'YOU' if is_user else f'Team {current_owner}'}")

        if is_user:
            st.write("### 🟢 It's your turn!")
            
            # RECOMENDACIÓN (Usando FPTS)
            if not available_df.empty:
                best_available = available_df.iloc[0]
                st.success(f"💡 Recommended Pick: **{best_available['Player']}** ({best_available['Position']}) - Proj: {best_available['FPTS']}")
            
            # --- TABS DE POSICIÓN PARA DRAFTEAR ---
            st.write("Filter by Position:")
            pos_tabs = st.tabs(["ALL", "QB", "RB", "WR", "TE", "K", "DST"])
            positions = ["ALL", "QB", "RB", "WR", "TE", "K", "DST"]
            
            player_drafted = None # Variable temporal para controlar la acción
            
            for i, tab in enumerate(pos_tabs):
                with tab:
                    pos = positions[i]
                    # Filtrar dataframe por tab
                    if pos == "ALL":
                        tab_df = available_df
                    else:
                        tab_df = available_df[available_df['Position'] == pos]
                    
                    if tab_df.empty:
                        st.warning(f"No {pos} available.")
                    else:
                        # Crear opciones para el selectbox
                        options = tab_df.apply(lambda x: f"{x['Player']} | {x['Position']} | {x['FPTS']} pts | {x['FPTS/G']} pts/g", axis=1)
                        # Clave única por tab
                        selection = st.selectbox(f"Select {pos}", options, key=f"sel_{pos}")
                        
                        if st.button(f"Draft Player ({pos})", key=f"btn_{pos}"):
                            player_drafted = selection.split(" | ")[0]
            
            # Ejecutar lógica de draft fuera de los tabs si se seleccionó alguien
            if player_drafted:
                player_data = df[df['Player'] == player_drafted].iloc[0].to_dict()
                
                # Guardar en mi equipo
                st.session_state.draft_state['my_team'].append(player_data)
                st.session_state.draft_state['drafted_players'].add(player_drafted)
                
                # Log
                st.session_state.draft_state['draft_log'].append({
                    'pick': st.session_state.draft_state['current_pick_overall'],
                    'round': current_round,
                    'team': 'My Team',
                    'player': player_drafted,
                    'pos': player_data['Position'],
                    'fpts': player_data['FPTS'],
                    'fpts_g': player_data['FPTS/G']
                })
                
                # Avanzar pick
                st.session_state.draft_state['current_pick_overall'] += 1
                st.rerun()
                
        else:
            limit_pick = st.session_state.draft_state['total_teams'] * st.session_state.draft_state['roster_size']
            if st.session_state.draft_state['current_pick_overall'] > limit_pick:
                 st.session_state.draft_state['finished'] = True
                 st.rerun()
            else:
                 st.warning("Simulating CPU picks...")
                 
    elif st.session_state.draft_state['finished']:
        st.success("🎉 Draft Completed!")

    # Draft History (Últimos 5)
    st.write("---")
    st.subheader("Draft Feed")
    if st.session_state.draft_state['draft_log']:
        log_df = pd.DataFrame(st.session_state.draft_state['draft_log'])
        
        # Renombrar columnas para visualización amigable
        # Usamos FPTS para 'Proj' y FPTS/G para 'Proj/G'
        display_df = log_df[['pick', 'team', 'player', 'pos', 'fpts', 'fpts_g']].copy()
        display_df = display_df.rename(columns={
            'pick': 'Pick',
            'team': 'Team',
            'player': 'Player',
            'pos': 'Pos',
            'fpts': 'Proj (Total)',
            'fpts_g': 'Proj/G'
        })
        
        st.dataframe(display_df.sort_values('Pick', ascending=False).head(10), hide_index=True, use_container_width=True)

# --- TAB 2: OPTIMIZER ---
with tab2:
    st.header("Lineup Optimizer")
    
    my_team = st.session_state.draft_state['my_team']
    
    if not my_team:
        st.warning("You haven't drafted any players yet. Go to the Draft Room.")
    else:
        st.write(f"Total Players on Roster: {len(my_team)}")
        
        col_opt1, col_opt2 = st.columns([1, 4])
        with col_opt1:
            optimize_btn = st.button("🚀 Optimize Lineup")
            
        starters, bench = optimize_lineup(my_team)
        
        # Calculo de puntos totales (FPTS)
        total_proj = sum([p['FPTS'] for p in starters])
        
        st.metric(label="Projected Total Points", value=f"{total_proj} pts")
        
        col1, col2 = st.columns(2)
        
        # Función auxiliar para formatear tablas
        def format_optimizer_table(data_list):
            if not data_list:
                return pd.DataFrame()
            temp_df = pd.DataFrame(data_list)
            # Asegurar que existan las columnas, si no, rellenar
            if 'FPTS/G' not in temp_df.columns:
                 temp_df['FPTS/G'] = 0
            
            # Seleccionar y renombrar
            final_df = temp_df[['Position', 'Player', 'FPTS', 'FPTS/G']].copy()
            final_df = final_df.rename(columns={
                'FPTS': 'Proj',
                'FPTS/G': 'Proj/G'
            })
            return final_df

        with col1:
            st.subheader("Starting Lineup (9)")
            if optimize_btn or True: 
                df_starters = format_optimizer_table(starters)
                st.dataframe(df_starters, hide_index=True)
                
        with col2:
            st.subheader("Bench (5)")
            if bench:
                df_bench = format_optimizer_table(bench)
                st.dataframe(df_bench, hide_index=True)
            else:
                st.write("No players on bench.")

# --- TAB 3: TRADE ANALYZER ---
with tab3:
    st.header("⚖️ Trade Analyzer")
    st.write("Select players to trade away and receive to see if it's fair.")
    
    my_players = [p['Player'] for p in st.session_state.draft_state['my_team']]
    all_players = df['Player'].tolist()
    others_players = [p for p in all_players if p not in my_players]
    
    col_give, col_get = st.columns(2)
    
    with col_give:
        st.subheader("You Give 📤")
        give_picks = st.multiselect("Select your players:", my_players)
        
    with col_get:
        st.subheader("You Receive 📥")
        get_picks = st.multiselect("Select players to acquire:", others_players)
        
    if give_picks and get_picks:
        # Calcular valores con FPTS
        val_give = df[df['Player'].isin(give_picks)]['FPTS'].sum()
        val_get = df[df['Player'].isin(get_picks)]['FPTS'].sum()
        
        diff = val_get - val_give
        
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Giving Value", f"{val_give} pts")
        c2.metric("Receiving Value", f"{val_get} pts")
        c3.metric("Net Difference", f"{diff:+.2f} pts", delta_color="normal")
        
        st.subheader("Verdict:")
        if diff > 20:
            st.success("🔥 YOU'RE ROBBING THEM! (Huge Win)")
            st.balloons()
        elif diff > 5:
            st.success("✅ Good Trade (You win slightly)")
        elif diff >= -5:
            st.info("⚖️ Fair Trade")
        elif diff >= -20:
            st.warning("⚠️ You're losing value")
        else:
            st.error("🚨 THEY'RE ROBBING YOU! (Bad Trade)")
    else:
        st.info("Select players on both sides to analyze.")

# --- TAB 4: POWER RANKINGS ---
with tab4:
    st.header("🏆 Power Rankings")
    st.write("Total projected points per team.")
    
    draft_log = st.session_state.draft_state['draft_log']
    
    if not draft_log:
        st.warning("Draft hasn't started yet.")
    else:
        rankings = calculate_power_rankings(draft_log, st.session_state.draft_state['total_teams'])
        
        if not rankings.empty:
            rankings_display = rankings.rename(columns={'team': 'Team', 'fpts': 'Total Proj Points'})
            
            st.dataframe(
                rankings_display,
                column_config={
                    "Total Proj Points": st.column_config.NumberColumn(
                        "Total FPTS",
                        format="%.0f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.write("Waiting for picks...")

