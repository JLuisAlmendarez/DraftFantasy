# Draft Fantasy
___
El siguiente proyecto esta pensado para ser desplegado en un servidor a traves de docker-compose.yml. Especificamente una instancia de EC2 en AWS. Los pasos son los siguientes.

1. Iniciar una instancia de EC2 (Liberar puerto 8000 y 8501, apartar al menos 15 gib, OS: ubuntu)
2. Ingresar a la computadora a traves de tu llave .pem previamente descargada o traves de la consola de aws.
3. git clone (a este repositorio)
4. Descargar docker-compose

Para descargar docker-compose ejecutar los siguientes comandos en orden.

* sudo apt update
* sudo apt upgrade -y
* sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
* curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
* sudo apt update
* sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
* sudo systemctl start docker
* sudo systemctl enable docker
* sudo usermod -aG docker $USER
* exit

Despues vuelves a acceder a la maquina, eso significa que se ha reiniciado.

5. Acceder a DraftFantasy/src
6. Crear tu archivo .env

Para crear tu archivo .env sigue las siguientes instrucciones y comandos

* nano .env
* escribir DATABRICKS_HOST=tu_url /parse DATABRICKS_TOKEN=tu_token
* ctrl + x
* Guardar

7. Ejecutar docker-compose up --build
8. Esperar a que los contenedores se levantes
9. Acceder a tus puertos a traves de la ip publica de AWS

____

Si se quiere entrenar un nuevo modelo, simplemente desde tu local o desde aws con las librerias instaladas.

1. Remplazar .env por tu clave correspondiente
2. python.exe DraftFantasy/src/pipelines/train_pipeline.py
