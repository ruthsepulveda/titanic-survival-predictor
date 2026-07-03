import streamlit as st
import numpy as np
import keras
import joblib

# ── Configuración de la página ────────────────────────────────────────────────
st.set_page_config(
    page_title="RMS Titanic — Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# ── CSS global para tipografía refinada ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IM+Fell+English:ital@0;1&display=swap');
    
    .refined-title {
        font-family: 'Playfair Display', Georgia, serif;
        letter-spacing: 6px;
        text-transform: uppercase;
    }
    .ticket-font {
        font-family: 'IM Fell English', 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── Cargar modelo y scaler ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return keras.models.load_model("titanic_model.keras")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ── Imágenes ──────────────────────────────────────────────────────────────────
IMAGES = {
    "titanic": "images/titanic.jpg",
    "first_class": "images/first_class.jpg",
    "second_class": "images/second_class.jpg",
    "third_class": "images/third_class.jpg",
    "lifeboats": "images/lifeboats.jpg",
    "sinking": "images/sinking.jpg"
}

def show_image(path, caption=""):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except Exception:
        pass

# ── Tarifas históricas ────────────────────────────────────────────────────────
FARES = {
    1: {
        "£30 — Camarote económico de primera clase": 30.0,
        "£75 — Camarote estándar de primera clase": 75.0,
        "£150 — Camarote de lujo de primera clase": 150.0,
        "£512 — Suite privada de primera clase": 512.0
    },
    2: {
        "£10 — Camarote económico de segunda clase": 10.0,
        "£15 — Camarote estándar de segunda clase": 15.0,
        "£25 — Camarote premium de segunda clase": 25.0
    },
    3: {
        "£5 — Litera económica de tercera clase": 5.0,
        "£8 — Litera estándar de tercera clase": 8.0,
        "£12 — Litera premium de tercera clase": 12.0
    }
}

CLASS_LABELS = {1: "Primera clase", 2: "Segunda clase", 3: "Tercera clase"}
PORT_LABELS = {
    "Southampton": "Southampton, Inglaterra",
    "Cherbourg": "Cherbourg, Francia",
    "Queenstown": "Queenstown, Irlanda"
}

# ── Narrativa personalizada ───────────────────────────────────────────────────
def get_narrative(survived, nombre, pclass, sex_label, age,
                  embarked_label, alone, family_size, fare_label):

    nombre_str = nombre.strip() if nombre.strip() else "Pasajero"
    clase_str = CLASS_LABELS[pclass].lower()
    puerto_str = PORT_LABELS[embarked_label]
    tarifa_str = fare_label.split("—")[0].strip()

    if alone:
        compania_str = "viajando solo"
    elif family_size == 1:
        compania_str = "junto a 1 familiar"
    else:
        compania_str = f"junto a {family_size} familiares"

    if survived:
        if sex_label == "Mujer":
            if pclass == 1:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. En la madrugada "
                    f"del 15 de abril de 1912, cuando el RMS Titanic comenzó a hundirse, "
                    f"tu posición en las cubiertas superiores y la política de mujeres y "
                    f"niños primero te dieron acceso prioritario a los botes salvavidas. "
                    f"A tus {age} años, lograste subir a uno de los 20 botes que zarparon "
                    f"esa noche. Las mujeres de primera clase tuvieron una tasa de "
                    f"supervivencia del 97%. Fuiste una de las afortunadas."
                )
            elif pclass == 2:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. Esa madrugada del "
                    f"15 de abril, a tus {age} años, la tripulación evacuó las cubiertas "
                    f"de segunda clase junto con las de primera. Lograste llegar a uno "
                    f"de los botes salvavidas antes de que el barco se hundiera "
                    f"completamente a las 2:20 AM. El 86% de las mujeres de segunda "
                    f"clase sobrevivieron esa noche."
                )
            else:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. Las escotillas de "
                    f"tercera clase fueron de las últimas en abrirse esa noche, pero "
                    f"a tus {age} años lograste abrirte camino hacia la cubierta "
                    f"superior a tiempo. Solo el 46% de las mujeres de tercera clase "
                    f"sobrevivieron. Tu determinación marcó la diferencia."
                )
        else:
            if pclass == 1:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. Como hombre de "
                    f"primera clase a tus {age} años, tu posición social te dio cierta "
                    f"ventaja en el caos de esa noche, pero los botes se llenaron "
                    f"principalmente con mujeres y niños. Lograste encontrar un lugar "
                    f"en uno de los últimos botes en zarpar. Solo el 34% de los hombres "
                    f"de primera clase sobrevivieron."
                )
            elif pclass == 2:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. A tus {age} años "
                    f"fuiste uno de los pocos hombres de segunda clase en sobrevivir "
                    f"esa noche. Solo el 8% de los hombres de tu clase lograron llegar "
                    f"a un bote salvavidas. Tu supervivencia fue extraordinaria."
                )
            else:
                if alone:
                    return (
                        f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                        f"{tarifa_str} de {clase_str}, viajando solo. A tus {age} años "
                        f"y sin familia que te retuviera, pudiste moverte más rápido "
                        f"que otros pasajeros de tercera clase. Lograste encontrar tu "
                        f"camino hacia la cubierta a tiempo. Solo el 17% de los hombres "
                        f"de tercera clase sobrevivieron esa noche."
                    )
                else:
                    return (
                        f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                        f"{tarifa_str} de {clase_str}, {compania_str}. A tus {age} "
                        f"años y viajando con familia, lograste guiar a los tuyos "
                        f"hacia la cubierta a tiempo. Solo el 17% de los hombres de "
                        f"tercera clase sobrevivieron. Tu historia es una de las más "
                        f"extraordinarias de esa noche."
                    )
    else:
        if pclass == 1:
            return (
                f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                f"{tarifa_str} de {clase_str}, {compania_str}. A tus {age} años, "
                f"ni tu posición privilegiada en las cubiertas superiores pudo "
                f"salvarte del caos de esa noche. El RMS Titanic se hundió a las "
                f"2:20 AM del 15 de abril de 1912 en el Atlántico Norte. Las aguas "
                f"alcanzaban los -2°C. De los 2,224 a bordo, 1,514 no sobrevivieron."
            )
        elif pclass == 2:
            return (
                f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                f"{tarifa_str} de {clase_str}, {compania_str}. A tus {age} años, "
                f"los pasillos de segunda clase se llenaron de confusión esa "
                f"madrugada. Los botes salvavidas se agotaron antes de que pudieras "
                f"llegar a cubierta. El Titanic se hundió a las 2:20 AM. El Carpathia "
                f"llegó recién a las 4:10 AM, demasiado tarde."
            )
        else:
            if alone:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, viajando solo. A tus {age} años, "
                    f"las escotillas de tercera clase permanecieron cerradas durante "
                    f"demasiado tiempo esa noche. Para cuando lograste llegar a la "
                    f"cubierta superior, los botes salvavidas ya habían partido. "
                    f"Solo el 25% de los pasajeros de tercera clase sobrevivieron."
                )
            else:
                return (
                    f"{nombre_str}, embarcaste en {puerto_str} con un boleto de "
                    f"{tarifa_str} de {clase_str}, {compania_str}. A tus {age} años, "
                    f"no quisiste dejar atrás a tus {family_size} familiar(es). "
                    f"Para cuando intentaron llegar juntos a cubierta, las escotillas "
                    f"de tercera clase ya llevaban demasiado tiempo bloqueadas. "
                    f"Los botes habían partido. Solo el 25% de los pasajeros de "
                    f"tercera clase sobrevivieron esa noche."
                )

# ── Inicializar session_state ─────────────────────────────────────────────────
if "show_result" not in st.session_state:
    st.session_state.show_result = False
if "current_pclass" not in st.session_state:
    st.session_state.current_pclass = 1

# ── Encabezado ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(180deg, #0a0a12 0%, #12121e 60%, #1a1520 100%);
    border: 1px solid #8b6914;
    border-radius: 2px;
    padding: 48px 40px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
">
    <div style="
        position: absolute;
        top: 12px; left: 12px; right: 12px; bottom: 12px;
        border: 1px solid #5a4510;
        border-radius: 1px;
        pointer-events: none;
    "></div>
    <p style="
        color: #c9a84c;
        font-size: 10px;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-bottom: 6px;
        font-family: 'Playfair Display', Georgia, serif;
    ">✦ White Star Line ✦</p>
    <h1 style="
        color: #f0e6c8;
        font-size: 48px;
        letter-spacing: 10px;
        text-transform: uppercase;
        margin: 8px 0;
        font-family: 'Playfair Display', Georgia, serif;
        font-weight: 700;
        text-shadow: 0 0 40px rgba(201, 168, 76, 0.3);
    ">RMS Titanic</h1>
    <div style="
        width: 120px;
        height: 1px;
        background: linear-gradient(90deg, transparent, #c9a84c, transparent);
        margin: 12px auto;
    "></div>
    <p style="
        color: #9a8a6a;
        font-size: 11px;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin: 0;
        font-family: Georgia, serif;
    ">Southampton — Nueva York</p>
    <p style="
        color: #5a4a3a;
        font-size: 10px;
        letter-spacing: 2px;
        margin-top: 8px;
        font-family: Georgia, serif;
    ">10 de abril de 1912 — Viaje inaugural</p>
</div>
""", unsafe_allow_html=True)

show_image(IMAGES["titanic"],
           "RMS Titanic partiendo de Southampton, 10 de abril de 1912")

st.markdown("""
<div style="
    text-align: center;
    color: #5a4a3a;
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 24px 0;
    font-family: Georgia, serif;
">✦ ✦ ✦</div>
""", unsafe_allow_html=True)

# ── Formulario ────────────────────────────────────────────────────────────────
if not st.session_state.show_result:

    st.markdown("""
    <p style="
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 22px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 4px;
    ">Reserva tu pasaje</p>
    <p style="
        color: #888888;
        font-size: 12px;
        letter-spacing: 1px;
        margin-bottom: 24px;
        font-family: Georgia, serif;
        font-style: italic;
    ">Complete su información para descubrir si habría sobrevivido al hundimiento del RMS Titanic</p>
    """, unsafe_allow_html=True)

    nombre = st.text_input(
        "Nombre del pasajero",
        placeholder="Escriba su nombre completo",
        key="nombre_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox(
            "Clase del pasaje",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Primera clase",
                2: "Segunda clase",
                3: "Tercera clase"
            }[x],
            key="pclass_input"
        )
        if pclass != st.session_state.current_pclass:
            st.session_state.current_pclass = pclass
            st.rerun()

    with col2:
        sex_label = st.radio(
            "Sexo",
            options=["Hombre", "Mujer"],
            horizontal=True,
            key="sex_input"
        )

    class_images = {1: "first_class", 2: "second_class", 3: "third_class"}
    class_captions = {
        1: "Comedor de primera clase del RMS Titanic",
        2: "Comedor de segunda clase del RMS Titanic",
        3: "Zona de tercera clase del RMS Titanic"
    }
    show_image(
        IMAGES[class_images[st.session_state.current_pclass]],
        class_captions[st.session_state.current_pclass]
    )

    col3, col4 = st.columns(2)
    with col3:
        age = st.number_input(
            "Edad", min_value=1, max_value=80, value=30, step=1,
            key="age_input"
        )
    with col4:
        embarked_label = st.selectbox(
            "Puerto de embarque",
            options=["Southampton", "Cherbourg", "Queenstown"],
            help="Southampton (Inglaterra), Cherbourg (Francia), Queenstown (Irlanda)",
            key="embarked_input"
        )

    col5, col6 = st.columns(2)
    with col5:
        alone = st.toggle(
            "Viajaba solo", value=False, key="alone_input"
        )
    with col6:
        family_size = st.number_input(
            "Familiares a bordo",
            min_value=1, max_value=10, value=1, step=1,
            disabled=alone, key="family_input"
        )

    fare_label = st.selectbox(
        "Precio del pasaje",
        options=list(FARES[st.session_state.current_pclass].keys()),
        help="Tarifas en libras esterlinas de 1912",
        key="fare_input"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("submit_form"):
        submitted = st.form_submit_button(
            "✦  Subir a bordo  ✦",
            use_container_width=True
        )
        if submitted:
            st.session_state.nombre = nombre
            st.session_state.pclass = pclass
            st.session_state.sex_label = sex_label
            st.session_state.age = age
            st.session_state.embarked_label = embarked_label
            st.session_state.alone = alone
            st.session_state.family_size = 0 if alone else family_size
            st.session_state.fare_label = fare_label
            st.session_state.show_result = True
            st.rerun()

# ── Resultado ─────────────────────────────────────────────────────────────────
else:
    nombre = st.session_state.nombre
    pclass = st.session_state.pclass
    sex_label = st.session_state.sex_label
    age = st.session_state.age
    embarked_label = st.session_state.embarked_label
    alone = st.session_state.alone
    family_size = st.session_state.family_size
    fare_label = st.session_state.fare_label

    sex = 0 if sex_label == "Hombre" else 1
    embarked = {"Southampton": 0, "Cherbourg": 1, "Queenstown": 2}[embarked_label]
    fare = FARES[pclass][fare_label]
    alone_int = 1 if alone else 0
    nombre_display = nombre.strip().upper() if nombre.strip() else "PASAJERO"
    tarifa_corta = fare_label.split("—")[0].strip()

    features = np.array([[pclass, sex, age, fare, embarked, alone_int, family_size]])
    features_scaled = scaler.transform(features)
    probability = float(model.predict(features_scaled, verbose=0)[0][0])
    survived = probability >= 0.5

    narrative = get_narrative(
        survived, nombre, pclass, sex_label, age,
        embarked_label, alone, family_size, fare_label
    )

    # ── Boleto de embarque ────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(160deg, #faf6ed 0%, #f0e8d0 100%);
        border: 1px solid #c8b89a;
        border-left: 6px solid #8b6914;
        border-radius: 2px;
        padding: 32px 40px 24px 40px;
        margin-bottom: 8px;
        position: relative;
    ">
        <div style="
            position: absolute;
            top: 8px; left: 14px; right: 8px; bottom: 8px;
            border: 1px solid rgba(139, 105, 20, 0.2);
            pointer-events: none;
        "></div>
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="
                color: #8b6914;
                font-size: 10px;
                letter-spacing: 5px;
                text-transform: uppercase;
                margin: 0 0 4px 0;
                font-family: Georgia, serif;
            ">✦ White Star Line ✦</p>
            <p style="
                color: #1a1208;
                font-size: 24px;
                font-weight: bold;
                letter-spacing: 6px;
                text-transform: uppercase;
                margin: 4px 0;
                font-family: 'Playfair Display', Georgia, serif;
            ">RMS Titanic</p>
            <p style="
                color: #6a5a3a;
                font-size: 10px;
                letter-spacing: 3px;
                text-transform: uppercase;
                margin: 0;
                font-family: Georgia, serif;
            ">Boleto de pasaje — Abril 1912</p>
            <div style="
                width: 80px;
                height: 1px;
                background: linear-gradient(90deg, transparent, #8b6914, transparent);
                margin: 12px auto 0;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    fields = [
        ("Pasajero", nombre_display),
        ("Clase", CLASS_LABELS[pclass]),
        ("Edad", f"{age} años"),
        ("Embarcó en", PORT_LABELS[embarked_label]),
        ("Tarifa", tarifa_corta),
        ("Viaja", "Solo" if alone else f"Con {family_size} familiar(es)"),
        ("Destino", "Nueva York, EE.UU."),
        ("Partida", "10 de abril de 1912"),
    ]

    for label, value in fields:
        col_label, col_value = st.columns([2, 3])
        with col_label:
            st.markdown(
                f"<p style='color: #8b6914; font-size: 11px; "
                f"letter-spacing: 2px; text-transform: uppercase; "
                f"margin: 3px 0; font-family: Georgia, serif;'>{label}</p>",
                unsafe_allow_html=True
            )
        with col_value:
            st.markdown(
                f"<p style='font-size: 13px; font-weight: bold; "
                f"letter-spacing: 1px; margin: 3px 0; "
                f"font-family: Georgia, serif;'>{value}</p>",
                unsafe_allow_html=True
            )

    st.markdown("""
    <div style="
        border-top: 1px dashed #c8b89a;
        margin-top: 20px;
        padding-top: 12px;
        text-align: center;
        color: #9a8060;
        font-size: 9px;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-family: Georgia, serif;
    ">Southampton &nbsp;✦&nbsp; Cherburgo &nbsp;✦&nbsp; Queenstown &nbsp;✦&nbsp; Nueva York</div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Separador estilo corte de boleto ──────────────────────────────────────
    st.markdown("""
    <div style="
        display: flex;
        align-items: center;
        margin: 8px 0 24px 0;
    ">
        <div style="flex: 1; border-top: 2px dashed #3a3030;"></div>
        <div style="
            margin: 0 16px;
            color: #5a4a3a;
            font-size: 14px;
            letter-spacing: 2px;
        ">✂</div>
        <div style="flex: 1; border-top: 2px dashed #3a3030;"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tarjeta de resultado ──────────────────────────────────────────────────
    if survived:
        border_color = "#c9a84c"
        result_text = "Sobreviviste"
        result_color = "#c9a84c"
        background = "linear-gradient(180deg, #0a0f0a 0%, #0a1a10 100%)"
        result_image = IMAGES["lifeboats"]
        result_caption = ("Supervivientes del Titanic fotografiados desde "
                          "el RMS Carpathia, 15 de abril de 1912")
        seal_color = "#c9a84c"
        seal_text = "✦"
    else:
        border_color = "#8b3333"
        result_text = "No sobreviviste"
        result_color = "#cc6666"
        background = "linear-gradient(180deg, #0f0808 0%, #1a0a0a 100%)"
        result_image = IMAGES["sinking"]
        result_caption = ("El hundimiento del RMS Titanic — "
                          "ilustración de Willy Stöwer, 1912")
        seal_color = "#8b3333"
        seal_text = "✝"

    st.markdown(
        f"""
        <div style="
            background: {background};
            border: 1px solid {border_color};
            border-radius: 2px;
            padding: 48px 40px;
            text-align: center;
            margin-bottom: 24px;
            position: relative;
        ">
            <div style="
                position: absolute;
                top: 10px; left: 10px; right: 10px; bottom: 10px;
                border: 1px solid rgba(201, 168, 76, 0.15);
                pointer-events: none;
            "></div>
            <p style="
                color: #5a4a3a;
                font-size: 9px;
                letter-spacing: 4px;
                text-transform: uppercase;
                margin-bottom: 20px;
                font-family: Georgia, serif;
            ">RMS Titanic &nbsp;✦&nbsp; 15 de abril de 1912 &nbsp;✦&nbsp; 2:20 AM</p>
            <div style="
                font-size: 32px;
                color: {seal_color};
                margin-bottom: 12px;
            ">{seal_text}</div>
            <p style="
                color: {result_color};
                font-size: 44px;
                font-weight: 700;
                letter-spacing: 6px;
                text-transform: uppercase;
                margin-bottom: 8px;
                font-family: 'Playfair Display', Georgia, serif;
                text-shadow: 0 0 30px rgba(201, 168, 76, 0.2);
            ">{result_text}</p>
            <div style="
                width: 80px;
                height: 1px;
                background: linear-gradient(90deg, transparent, {border_color}, transparent);
                margin: 16px auto;
            "></div>
            <p style="
                color: #6a5a4a;
                font-size: 11px;
                letter-spacing: 2px;
                margin-bottom: 28px;
                font-family: Georgia, serif;
                font-style: italic;
            ">Probabilidad de supervivencia: {probability*100:.1f}%</p>
            <p style="
                color: #c8baa8;
                font-size: 15px;
                line-height: 2.0;
                max-width: 520px;
                margin: 0 auto;
                font-style: italic;
                font-family: 'Playfair Display', Georgia, serif;
            ">{narrative}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_image(result_image, result_caption)

    st.markdown("""
    <div style="
        text-align: center;
        color: #5a4a3a;
        font-size: 10px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin: 16px 0;
        font-family: Georgia, serif;
    ">✦ ✦ ✦</div>
    """, unsafe_allow_html=True)

    st.progress(int(probability * 100))
    st.caption(f"Probabilidad de supervivencia: {probability*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "✦  Intentar con otro pasajero  ✦",
        use_container_width=True,
        key="btn_retry"
    ):
        st.session_state.show_result = False
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Contexto histórico"):
        st.markdown("""
        ### Tasas de supervivencia históricas

        | Grupo | Sobrevivieron |
        |---|---|
        | Mujeres de 1ra clase | 97% |
        | Mujeres de 2da clase | 86% |
        | Mujeres de 3ra clase | 46% |
        | Hombres de 1ra clase | 34% |
        | Hombres de 2da clase | 8% |
        | Hombres de 3ra clase | 17% |
        | Niños menores de 15 | 52% |

        ### El hundimiento
        El RMS Titanic chocó con un iceberg a las 23:40 del 14 de abril
        de 1912 y se hundió a las 2:20 del 15 de abril. De los 2,224
        pasajeros y tripulantes a bordo, solo 710 sobrevivieron. El barco
        contaba con botes salvavidas para apenas 1,178 personas. Las aguas
        del Atlántico Norte alcanzaban los -2°C esa noche.

        ### Sobre el modelo
        Este predictor usa una red neuronal MLP entrenada con datos reales
        de los pasajeros del Titanic. El modelo tiene un accuracy del 79.7%
        en datos de prueba, lo que refleja que la supervivencia tuvo un
        componente de azar que ningún modelo puede capturar completamente.
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    text-align: center;
    color: #3a3030;
    font-size: 9px;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 32px;
    font-family: Georgia, serif;
">
    ✦ &nbsp; Proyecto de portafolio &nbsp; ✦ &nbsp;
    Red neuronal MLP entrenada con Keras &nbsp; ✦ &nbsp;
    Dataset del Titanic (Seaborn) &nbsp; ✦
</div>
""", unsafe_allow_html=True)

