import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore", category=UserWarning)

import streamlit as st
import whisper
import tempfile
from audio_recorder_streamlit import audio_recorder
import re
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Asistente Médico por Voz",
    page_icon="🩺",
    layout="wide"
)

# Base de conocimientos médica
MEDICAL_KNOWLEDGE = {
    "fiebre": {
        "keywords": ["fiebre", "temperatura", "hipertermia", "febril", "caliente"],
        "normal_range": (36.0, 37.5),
        "severity": {
            "leve": (37.5, 38.5),
            "moderada": (38.5, 39.5), 
            "alta": (39.5, 42.0),
            "critica": (42.0, 50.0)
        }
    },
    "respiracion": {
        "keywords": ["respirar", "disnea", "ahogo", "falta aire", "sofoco", "asfixia", "no puede respirar"],
        "conditions": ["asma", "epoc", "neumonia", "embolia"]
    },
    "dolor_pulmon": {
        "keywords": ["dolor pulmon", "dolor pecho", "dolor torax", "opresion", "punzada"],
        "conditions": ["neumonia", "pleuritis", "embolia", "neumotorax"]
    },
    "tos": {
        "keywords": ["tos", "toser", "expectorar", "flema"],
        "conditions": ["bronquitis", "neumonia", "tuberculosis"]
    }
}

# Diagnósticos y recomendaciones NIC
DIAGNOSTICOS_NIC = {
    "neumonia": {
        "sintomas_clave": ["fiebre", "dolor_pulmon", "respiracion", "tos"],
        "recomendaciones": [
            "Control de temperatura: Administrar antipireticos segun prescripcion medica",
            "Hidratacion: Asegurar ingesta adecuada de liquidos 2-3L por dia",
            "Monitorizacion respiratoria: Vigilar frecuencia respiratoria y saturacion O2",
            "Administracion de antibioticos: Segun cultivo y antibiograma",
            "Posicionamiento: Semi-Fowler para facilitar ventilacion",
            "Prevencion infecciones: Lavado de manos y tecnicas asepticas",
            "Educacion: Enseñar tecnicas de respiracion profunda y tos efectiva"
        ],
        "nic_codes": ["NIC 3350 - Monitorizacion Respiratoria", "NIC 6550 - Proteccion Infecciones", "NIC 1100 - Manejo Nutricional"]
    },
    "asma_aguda": {
        "sintomas_clave": ["respiracion", "sibilancias", "opresion"],
        "recomendaciones": [
            "Broncodilatadores: Administrar salbutamol segun prescripcion",
            "Posicion: Mantener en posicion Fowler alta o tripode",
            "Tecnicas respiracion: Enseñar respiracion diafragmatica",
            "Identificar desencadenantes: Alejar alergenos conocidos",
            "Monitoreo: Vigilar peak flow y saturacion O2",
            "Ambiente: Mantener ambiente tranquilo libre de irritantes",
            "Signos alarma: Educar sobre cuando buscar ayuda urgente"
        ],
        "nic_codes": ["NIC 3140 - Manejo Via Aerea", "NIC 5602 - Enseñanza Proceso Enfermedad", "NIC 3350 - Monitorizacion Respiratoria"]
    },
    "crisis_respiratoria": {
        "sintomas_clave": ["respiracion", "ahogo"],
        "recomendaciones": [
            "PRIORIDAD: Evaluar via aerea, respiracion, circulacion (ABC)",
            "Oxigenoterapia: Administrar O2 segun saturacion y prescripcion",
            "Alertar medico: Comunicar inmediatamente cambios en estado",
            "Signos vitales: Monitorizar cada 15-30 minutos",
            "Posicion: Fowler alta o posicion de comodidad del paciente",
            "Calmar paciente: Reducir ansiedad con presencia y explicaciones",
            "Acceso vascular: Asegurar via IV permeable"
        ],
        "nic_codes": ["NIC 6320 - Reanimacion Cardiopulmonar", "NIC 3350 - Monitorizacion Respiratoria", "NIC 5820 - Disminucion Ansiedad"]
    }
}

def extraer_informacion_medica(texto):
    """Extrae informacion medica del texto hablado"""
    texto = texto.lower()
    
    # Extraer temperatura
    temp_patterns = [
        r"(\d+(?:\.\d+)?)\s*(?:grados?|°)",
        r"temperatura\s*(?:de\s*)?(\d+(?:\.\d+)?)",
        r"fiebre\s*(?:de\s*)?(\d+(?:\.\d+)?)"
    ]
    
    temperatura = None
    for pattern in temp_patterns:
        match = re.search(pattern, texto)
        if match:
            temperatura = float(match.group(1))
            break
    
    # Identificar sintomas presentes
    sintomas_detectados = []
    
    for sintoma, data in MEDICAL_KNOWLEDGE.items():
        for keyword in data["keywords"]:
            if keyword in texto:
                sintomas_detectados.append(sintoma)
                break
    
    # Extraer informacion adicional
    info = {
        "temperatura": temperatura,
        "sintomas": sintomas_detectados,
        "texto_original": texto,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    
    return info

def generar_diagnostico_y_recomendaciones(info_medica):
    """Genera diagnostico probable y recomendaciones NIC"""
    
    sintomas = info_medica["sintomas"]
    temperatura = info_medica["temperatura"]
    
    # Logica de diagnostico
    diagnostico_probable = None
    urgencia = "normal"
    
    # Evaluar severidad por temperatura
    if temperatura:
        if temperatura >= 42:
            urgencia = "critica"
        elif temperatura >= 39.5:
            urgencia = "alta"
        elif temperatura >= 38.5:
            urgencia = "moderada"
    
    # Determinar diagnostico probable
    if "fiebre" in sintomas and "dolor_pulmon" in sintomas and "respiracion" in sintomas:
        diagnostico_probable = "neumonia"
    elif "respiracion" in sintomas and temperatura and temperatura < 38:
        diagnostico_probable = "asma_aguda"
    elif "respiracion" in sintomas:
        diagnostico_probable = "crisis_respiratoria"
        urgencia = "alta"
    
    # Generar recomendaciones
    recomendaciones = []
    nic_codes = []
    
    if diagnostico_probable and diagnostico_probable in DIAGNOSTICOS_NIC:
        data = DIAGNOSTICOS_NIC[diagnostico_probable]
        recomendaciones = data["recomendaciones"]
        nic_codes = data["nic_codes"]
    
    # Recomendaciones especificas por temperatura
    if temperatura:
        if temperatura >= 39:
            recomendaciones.insert(0, f"FIEBRE ALTA ({temperatura}°C): Medidas fisicas de enfriamiento + antipireticos URGENTE")
        elif temperatura >= 38:
            recomendaciones.insert(0, f"Fiebre moderada ({temperatura}°C): Control cada 4 horas, antipireticos segun prescripcion")
    
    return {
        "diagnostico": diagnostico_probable,
        "urgencia": urgencia,
        "recomendaciones": recomendaciones,
        "nic_codes": nic_codes,
        "sintomas_detectados": sintomas
    }

# Inicializar session state
if "historial_pacientes" not in st.session_state:
    st.session_state.historial_pacientes = []

if "whisper_model" not in st.session_state:
    st.session_state.whisper_model = None

# Titulo principal
st.title("Asistente Medico por Voz")
st.markdown("**Describe a tu paciente hablando y recibe recomendaciones NIC inmediatas**")


# Columnas principales
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Describe al Paciente")
    
    # Grabacion de audio
    st.info("Presiona el boton y describe los sintomas de tu paciente")
    audio_bytes = audio_recorder(
        text="Presiona para describir al paciente",
        recording_color="#e74c3c",
        neutral_color="#2ecc71",
        icon_name="microphone",
        icon_size="2x"
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
    
    # Opcion de texto directo
    st.markdown("**O escribe directamente:**")
    texto_manual = st.text_area(
        "Describe los sintomas:",
        placeholder="Ej: El paciente tiene fiebre de 39 grados y no puede respirar",
        height=100
    )

with col2:
    st.subheader("Analisis y Recomendaciones")
    
    # Boton de analisis
    if st.button("Analizar Paciente", use_container_width=True, key="analizar_btn"):
        texto_analizar = None
        
        # Procesar audio si existe
        if audio_bytes:
            # Cargar modelo Whisper (solo una vez)
            if st.session_state.whisper_model is None:
                with st.spinner("Cargando modelo de reconocimiento de voz..."):
                    try:
                        model = whisper.load_model("base", device="cpu")
                        st.session_state.whisper_model = model
                    except Exception as e:
                        st.error(f"Error al cargar modelo: {str(e)}")
                        st.stop()
            else:
                model = st.session_state.whisper_model
            
            # Transcribir audio
            with st.spinner("Transcribiendo descripcion del paciente..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    result = model.transcribe(tmp_path)
                    texto_analizar = result["text"].strip()
                    os.unlink(tmp_path)
                    
                    st.success(f"Transcripcion: {texto_analizar}")
                    
                except Exception as e:
                    st.error(f"Error en transcripcion: {str(e)}")
        
        # Usar texto manual si no hay audio
        elif texto_manual:
            texto_analizar = texto_manual
        
        # Analizar informacion medica
        if texto_analizar:
            with st.spinner("Analizando informacion medica..."):
                info_medica = extraer_informacion_medica(texto_analizar)
                resultado = generar_diagnostico_y_recomendaciones(info_medica)
                
                # Mostrar analisis
                st.success("Analisis completado")
                
                # Informacion detectada
                col_info1, col_info2 = st.columns(2)
                
                with col_info1:
                    if info_medica["temperatura"]:
                        temp = info_medica["temperatura"]
                        if temp >= 39:
                            st.error(f"Temperatura: {temp}°C (ALTA)")
                        elif temp >= 38:
                            st.warning(f"Temperatura: {temp}°C")
                        else:
                            st.info(f"Temperatura: {temp}°C")
                    else:
                        st.info("Temperatura: No especificada")
                
                with col_info2:
                    if resultado["urgencia"] == "critica":
                        st.error(f"Urgencia: {resultado['urgencia'].upper()}")
                    elif resultado["urgencia"] == "alta":
                        st.warning(f"Urgencia: {resultado['urgencia'].upper()}")
                    else:
                        st.info(f"Urgencia: {resultado['urgencia']}")
                
                # Sintomas detectados
                if resultado["sintomas_detectados"]:
                    st.write("**Sintomas identificados:**")
                    for sintoma in resultado["sintomas_detectados"]:
                        st.write(f"- {sintoma.replace('_', ' ').title()}")
                
                # Diagnostico probable
                if resultado["diagnostico"]:
                    st.write(f"**Diagnostico probable:** {resultado['diagnostico'].replace('_', ' ').title()}")
                
                # Codigos NIC
                if resultado["nic_codes"]:
                    st.write("**Codigos NIC aplicables:**")
                    for code in resultado["nic_codes"]:
                        st.write(f"- {code}")
                
                # Recomendaciones
                if resultado["recomendaciones"]:
                    st.write("**Recomendaciones de Enfermeria:**")
                    for i, recomendacion in enumerate(resultado["recomendaciones"], 1):
                        st.write(f"{i}. {recomendacion}")
                
                # Guardar en historial
                paciente_info = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "descripcion": texto_analizar,
                    "temperatura": info_medica["temperatura"],
                    "diagnostico": resultado["diagnostico"],
                    "urgencia": resultado["urgencia"],
                    "sintomas": resultado["sintomas_detectados"],
                    "recomendaciones": resultado["recomendaciones"]
                }
                
                st.session_state.historial_pacientes.append(paciente_info)

# Historial de pacientes
if st.session_state.historial_pacientes:
    st.header("Historial de Pacientes")
    
    if st.button("Limpiar Historial", key="limpiar_btn"):
        st.session_state.historial_pacientes = []
        st.rerun()
    
    for i, paciente in enumerate(reversed(st.session_state.historial_pacientes)):
        with st.expander(f"Paciente #{len(st.session_state.historial_pacientes) - i} - {paciente['timestamp']}"):
            st.write(f"**Descripcion:** {paciente['descripcion']}")
            st.write(f"**Temperatura:** {paciente['temperatura']}°C" if paciente['temperatura'] else "**Temperatura:** No especificada")
            st.write(f"**Diagnostico:** {paciente['diagnostico']}" if paciente['diagnostico'] else "**Diagnostico:** A determinar")
            st.write(f"**Urgencia:** {paciente['urgencia']}")
            
            if paciente.get("recomendaciones"):
                st.write("**Recomendaciones aplicadas:**")
                for rec in paciente["recomendaciones"][:3]:  # Mostrar solo las primeras 3
                    st.write(f"- {rec}")

# Footer
st.markdown("---")
st.markdown("**Importante:** Este es un sistema de apoyo. Siempre confirme con evaluacion medica profesional.")
