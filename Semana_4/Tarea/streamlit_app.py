"""
Laboratorio de Recomendación

El objetivo: una app donde alguien de negocio (que no sabe programar) pueda explorar sus recomendadores:
elegir un cliente y un método, ver sus 3 recomendaciones con el "porque", y comparar los métodos
(hit rate y cobertura) — con al menos un control interactivo.

Correr con:  streamlit run streamlit_app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Recomendador — Laboratorio", page_icon="☕", layout="wide")

VERDE = "#00704A"
NARANJA = "#C4512F"
GRIS = "#9A988E"

METODOS = ["Popularidad", "Filtrado por ítem", "Filtrado por cliente"]


def carpeta_datos():
    """Busca la carpeta data/ junto a este archivo, en la carpeta de trabajo o un nivel arriba."""
    aqui = Path(__file__).resolve().parent
    for base in (aqui, Path.cwd(), aqui.parent):
        if (base / "data" / "catalogo.csv").exists():
            return base / "data"
    st.error("No encuentro data/catalogo.csv. Sube la carpeta data/ al repositorio, junto a este archivo.")
    st.stop()


@st.cache_data
def cargar():
    path = carpeta_datos()
    catalogo = pd.read_csv(path / "catalogo.csv")
    hist = pd.read_csv(path / "ordenes_historial.csv")
    hold = pd.read_csv(path / "ordenes_holdout.csv")
    exp = pd.read_csv(path / "experimento.csv")
    clientes = pd.read_csv(path / "clientes.csv")
    productos = list(catalogo["item"])
    M = (pd.crosstab(hist["customer_id"], hist["item"]) > 0).astype(int).reindex(
        index=clientes["customer_id"], columns=productos, fill_value=0)
    X = M.values.astype(np.float32)
    H = (pd.crosstab(hold["customer_id"], hold["item"]) > 0).astype(int).reindex(
        index=M.index, columns=productos, fill_value=0).values
    NUEVO = H * (1 - X)                                   # probó algo NUEVO en el holdout
    NUEVO[:, productos.index("Agua")] = 0
    es_control = (exp.set_index("customer_id").loc[M.index, "grupo"] == "control").values
    return productos, M.index.values, X, NUEVO, es_control


productos, ids, X, NUEVO, es_control = cargar()
COL_AGUA = productos.index("Agua")
popularidad = X.sum(axis=0)


# TODO 1: la tabla de similitudes producto × producto (con la diagonal en 0), calculada UNA vez
@st.cache_data
def tabla_similitud_items(_X):
    S = cosine_similarity(_X.T)
    np.fill_diagonal(S, 0)
    return S


S_it = tabla_similitud_items(X)


def vecinos_de(x_vec, K, excluir=None):
    """Índices de los K clientes más parecidos a x_vec, con sus similitudes."""
    sims = cosine_similarity(x_vec[None, :], X)[0]
    if excluir is not None:
        sims[excluir] = -1
    idx = np.argsort(-sims)[:K]
    return idx, sims[idx]


def puntajes(metodo, x_vec, K, excluir=None):
    """Puntajes de los 29 productos para UN cliente (x_vec = su vector de 0/1)."""
    if metodo == "Popularidad":
        return popularidad.astype(float)
    if metodo == "Filtrado por ítem":
        # TODO 2: suma de las similitudes de cada producto con lo que el cliente ya compró
        return S_it @ x_vec
    # TODO 3: filtrado por cliente: similitud del cliente con TODOS los demás, sus K vecinos más parecidos
    #         y la suma de sus compras ponderada por esa similitud
    idx, sims = vecinos_de(x_vec, K, excluir)
    return sims @ X[idx]


def top3(scores, x_vec):
    s = np.where(x_vec > 0, -np.inf, scores).astype(float)
    s[COL_AGUA] = -np.inf
    return np.argsort(-s)[:3]


def porque(metodo, j, x_vec, K, excluir):
    """Explicación en lenguaje de negocio de por qué se recomienda el producto j."""
    if metodo == "Popularidad":
        return "es de lo más comprado por todos los clientes"
    if metodo == "Filtrado por ítem":
        comprados = np.where(x_vec > 0)[0]
        i = comprados[np.argmax(S_it[j, comprados])]
        return f"porque compró {productos[i]} (similitud {S_it[j, i]:.2f})"
    idx, _ = vecinos_de(x_vec, K, excluir)
    n = int(X[idx, j].sum())
    return f"{n} de sus {K} vecinos más parecidos lo compraron"


@st.cache_data
def evaluar(K, n_clientes, semilla=42):
    """Hit rate@3 y cobertura de cada método sobre clientes CONTROL que probaron algo nuevo."""
    elegibles = np.where((NUEVO.sum(axis=1) > 0) & es_control)[0]
    rng = np.random.default_rng(semilla)
    filas = rng.choice(elegibles, size=min(n_clientes, len(elegibles)), replace=False)
    Xs = X[filas]

    resultados = {}
    for metodo in METODOS:
        if metodo == "Popularidad":
            S = np.tile(popularidad.astype(float), (len(filas), 1))
        elif metodo == "Filtrado por ítem":
            S = Xs @ S_it.T
        else:
            sims = cosine_similarity(Xs, X)
            sims[np.arange(len(filas)), filas] = -1
            vec = np.argpartition(-sims, K - 1, axis=1)[:, :K]
            W = np.zeros_like(sims)
            np.put_along_axis(W, vec, np.take_along_axis(sims, vec, axis=1), axis=1)
            S = W @ X
        S = np.where(Xs > 0, -np.inf, S)
        S[:, COL_AGUA] = -np.inf
        recs = np.argsort(-S, axis=1)[:, :3]
        aciertos = np.take_along_axis(NUEVO[filas], recs, axis=1).sum(axis=1) > 0
        resultados[metodo] = {
            "Hit rate@3": aciertos.mean(),
            "Cobertura": len(np.unique(recs)) / (len(productos) - 1),
        }
    return pd.DataFrame(resultados).T, len(filas), len(elegibles)


# --- Título ---
st.title("☕ Recomendador de productos")
st.caption("Esta app ayuda a decidir 3 productos mostrar a cada cliente en el carrusel "
           "y con qué método de recomendación, comparando qué tan seguido aciertan.")

# TODO 4: barra lateral con los controles interactivos (mínimo UNO): un selector del método
#         (st.selectbox) y un slider con el número de vecinos K (st.slider)
with st.sidebar:
    st.header("Parámetros")
    metodo = st.selectbox("Método de recomendación", METODOS, index=1)
    K = st.slider("Número de vecinos K (filtrado por cliente)", 5, 100, 30, step=5,
                  disabled=(metodo != "Filtrado por cliente"),
                  help="Cuántos clientes parecidos se consultan. Solo aplica al filtrado por cliente.")

tab1, tab2 = st.tabs(["Recomendar", "Comparar métodos"])

with tab1:
    cid = st.number_input("ID de cliente", min_value=int(ids.min()), max_value=int(ids.max()), value=14, step=1)
    coincidencias = np.where(ids == cid)[0]
    if len(coincidencias) == 0:
        st.error("Ese ID de cliente no existe en la base.")
        st.stop()
    fila = int(coincidencias[0])
    x_vec = X[fila].copy()
    comprados = [productos[j] for j in np.where(x_vec > 0)[0]]
    st.write("Ya compró:", comprados if comprados else "nada antes del corte")

    # TODO 5: muestren las 3 recomendaciones con su puntaje y, para cada una, el "porque"
    #         - por ítem: "porque compró X (similitud 0.87)"
    #         - por cliente: "N de sus K vecinos más parecidos lo compraron"
    #         - popularidad: "es de lo más comprado"
    #         (y qué hacen con un cliente SIN historial: los métodos colaborativos dan puntaje 0)
    
    metodo_usado = metodo
    if x_vec.sum() == 0 and metodo != "Popularidad":
        st.warning("Este cliente no tiene historial: los métodos colaborativos dan puntaje 0. "
                   "Se usa Popularidad como respaldo (arranque en frío).")
        metodo_usado = "Popularidad"

    scores = puntajes(metodo_usado, x_vec, K, excluir=fila)
    recs = top3(scores, x_vec)

    st.subheader(f"Top 3 recomendaciones ({metodo_usado})")
    cols = st.columns(3)
    for pos, (col, j) in enumerate(zip(cols, recs), start=1):
        with col:
            st.metric(f"#{pos} {productos[j]}", f"{scores[j]:.2f}", help="Puntaje del método")
            texto = porque(metodo_usado, j, x_vec, K, fila)
            st.caption(texto[0].upper() + texto[1:])

with tab2:
    # TODO 6: tabla de comparación (hit rate@3 y cobertura de cada método) calculada sobre los clientes del grupo
    #         CONTROL que probaron algo nuevo (NUEVO.sum(axis=1) > 0 y es_control), con al menos un control
    #         interactivo (por ejemplo, un slider con cuántos clientes evaluar o el K del filtrado por cliente).
    #         Pista: es la misma lógica de hit_rate_at_k del notebook.
    n_elegibles = int(((NUEVO.sum(axis=1) > 0) & es_control).sum())
    n_eval = st.slider("Clientes del grupo control a evaluar", 100, max(n_elegibles, 101),
                       min(1000, n_elegibles), step=50)
    tabla, n_usados, n_total = evaluar(K, n_eval)
    st.caption(f"Evaluado sobre {n_usados:,} de {n_total:,} clientes de control que probaron algo nuevo "
               f"en los 90 días (K = {K} para el filtrado por cliente).")

    mostrar = tabla.copy()
    mostrar["Hit rate@3"] = mostrar["Hit rate@3"].map("{:.1%}".format)
    mostrar["Cobertura"] = mostrar["Cobertura"].map("{:.1%}".format)
    st.dataframe(mostrar)

    st.bar_chart(tabla["Hit rate@3"], color=VERDE)
    st.caption("Hit rate@3: % de clientes para quienes al menos 1 de las 3 recomendaciones fue un producto nuevo "
               "que sí probaron. Cobertura: % del catálogo (sin Agua) que el método llega a recomendar.")
