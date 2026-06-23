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



def desativar_morador(
    morador_id
    ):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE morador
        SET ativo = 0
        WHERE id = ?
    """,
    (morador_id,))

    conn.commit()
    conn.close()


def ativar_morador(
    morador_id
    ):
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE morador
        SET ativo = 1
        WHERE id = ?
    """,
    (morador_id,))

    conn.commit()
    conn.close()

def listar_moradores():

    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            id,
            nome,
            apartamento,
            bloco,
            ativo
        FROM morador
        ORDER BY nome
    """)

    moradores = cursor.fetchall()

    conn.close()

    return moradores


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

    morador = cursor.fetchone()

    conn.close()

    return morador



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

