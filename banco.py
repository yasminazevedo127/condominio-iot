import sqlite3

DB_PATH = "database/condominio.db"

def conectar():
    return sqlite3.connect(DB_PATH)

def inserir_morador(
    nome,
    apartamento,
    bloco,
    caminho_foto,
    caminho_embedding
    ):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO morador (
            nome,
            apartamento,
            bloco,
            caminho_foto,
            caminho_embedding
        )
        VALUES (?, ?, ?, ?, ?)
    """,
    (
        nome,
        apartamento,
        bloco,
        caminho_foto,
        caminho_embedding
    ))

    conn.commit()

    morador_id = cursor.lastrowid

    conn.close()

    return morador_id


def buscar_morador_por_id(
    morador_id
    ):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM morador
        WHERE id = ?
    """,
    (morador_id,))

    resultado = cursor.fetchone()

    conn.close()

    return resultado

def listar_moradores():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM morador
        WHERE ativo = 1
    """)

    moradores = cursor.fetchall()

    conn.close()

    return moradores


def registrar_acesso(
    morador_id,
    autorizado,
    distancia,
    data_hora
    ):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO acesso (
            morador_id,
            data_hora,
            autorizado,
            distancia
        )
        VALUES (?, ?, ?, ?)
    """,
    (
        morador_id,
        data_hora,
        autorizado,
        distancia
    ))

    conn.commit()
    conn.close()

