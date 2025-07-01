import streamlit as st
import re
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Asistente Médico por Texto",
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
    """Extrae informacion medica del texto"""
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

# Titulo principal
st.title("🩺 Asistente Médico por Texto")
st.markdown("**Describe a tu paciente y recibe recomendaciones NIC inmediatas**")

# Alerta sobre reconocimiento de voz
st.info("ℹ️ Temporalmente solo disponible entrada de texto. Reconocimiento de voz estará disponible próximamente.")

# Instrucciones
with st.expander("📋 Cómo usar el asistente"):
    st.markdown("""
    **Ejemplos de lo que puedes escribir:**
    - "Mi paciente tiene fiebre de 39 grados y no puede respirar bien"
    - "El paciente presenta dolor en el pulmon y tos con flema"
    - "Tiene temperatura de 38.5 y se ahoga al caminar"
    - "Fiebre alta de 40 grados, dolor toracico y disnea"
    
    **El sistema detectara automaticamente:**
    - Temperatura/fiebre
    - Problemas respiratorios
    - Dolor toracico/pulmonar
    - Otros sintomas relevantes
    """)

# Columnas principales
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Describe al Paciente")
    
    # Entrada de texto principal
    texto_manual = st.text_area(
        "Describe los sintomas del paciente:",
        placeholder="Ej: El paciente tiene fiebre de 39 grados y no puede respirar",
        height=150,
        help="Describe los sintomas de forma natural, incluyendo temperatura si la conoces"
    )
    
    # Datos adicionales opcionales
    with st.expander("➕ Datos adicionales (opcional)"):
        col_extra1, col_extra2 = st.columns(2)
        
        with col_extra1:
            edad = st.number_input("Edad del paciente", min_value=0, max_value=120, value=None)
            peso = st.number_input("Peso (kg)", min_value=0.0, value=None)
        
        with col_extra2:
            genero = st.selectbox("Género", ["No especificado", "Masculino", "Femenino", "Otro"])
            alergias = st.text_input("Alergias conocidas")

with col2:
    st.subheader("📊 Análisis y Recomendaciones")
    
    # Botón de análisis
    if st.button("🔍 Analizar Paciente", use_container_width=True, key="analizar_btn"):
        
        if not texto_manual.strip():
            st.warning("⚠️ Por favor, describe los síntomas del paciente.")
            st.stop()
        
        # Analizar información médica
        with st.spinner("Analizando información médica..."):
            info_medica = extraer_informacion_medica(texto_manual)
            resultado = generar_diagnostico_y_recomendaciones(info_medica)
            
            # Mostrar análisis
            st.success("✅ Análisis completado")
            
            # Información detectada
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if info_medica["temperatura"]:
                    temp = info_medica["temperatura"]
                    if temp >= 39:
                        st.error(f"🌡️ Temperatura: {temp}°C (ALTA)")
                    elif temp >= 38:
                        st.warning(f"🌡️ Temperatura: {temp}°C")
                    else:
                        st.info(f"🌡️ Temperatura: {temp}°C")
                else:
                    st.info("🌡️ Temperatura: No especificada")
            
            with col_info2:
                if resultado["urgencia"] == "critica":
                    st.error(f"⚠️ Urgencia: {resultado['urgencia'].upper()}")
                elif resultado["urgencia"] == "alta":
                    st.warning(f"⚠️ Urgencia: {resultado['urgencia'].upper()}")
                else:
                    st.info(f"ℹ️ Urgencia: {resultado['urgencia']}")
            
            # Síntomas detectados
            if resultado["sintomas_detectados"]:
                st.write("**🩺 Síntomas identificados:**")
                for sintoma in resultado["sintomas_detectados"]:
                    st.write(f"• {sintoma.replace('_', ' ').title()}")
            
            # Diagnóstico probable
            if resultado["diagnostico"]:
                st.write(f"**🎯 Diagnóstico probable:** {resultado['diagnostico'].replace('_', ' ').title()}")
            
            # Códigos NIC
            if resultado["nic_codes"]:
                st.write("**📋 Códigos NIC aplicables:**")
                for code in resultado["nic_codes"]:
                    st.write(f"• {code}")
            
            # Recomendaciones
            if resultado["recomendaciones"]:
                st.write("**💡 Recomendaciones de Enfermería:**")
                for i, recomendacion in enumerate(resultado["recomendaciones"], 1):
                    st.write(f"{i}. {recomendacion}")
            
            # Guardar en historial
            paciente_info = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "descripcion": texto_manual,
                "temperatura": info_medica["temperatura"],
                "diagnostico": resultado["diagnostico"],
                "urgencia": resultado["urgencia"],
                "sintomas": resultado["sintomas_detectados"],
                "recomendaciones": resultado["recomendaciones"],
                "edad": edad if 'edad' in locals() and edad else None,
                "genero": genero if 'genero' in locals() else None
            }
            
            st.session_state.historial_pacientes.append(paciente_info)
            
            # Mensaje de éxito
            st.success("✅ Información guardada en el historial")

# Botón para limpiar el texto
if texto_manual:
    if st.button("🗑️ Limpiar texto", key="limpiar_texto"):
        st.rerun()

# Historial de pacientes
if st.session_state.historial_pacientes:
    st.header("📋 Historial de Pacientes")
    
    col_hist1, col_hist2 = st.columns([3, 1])
    
    with col_hist2:
        if st.button("🗑️ Limpiar Historial", key="limpiar_btn"):
            st.session_state.historial_pacientes = []
            st.rerun()
    
    with col_hist1:
        st.write(f"Total de evaluaciones: {len(st.session_state.historial_pacientes)}")
    
    # Mostrar últimas 5 evaluaciones
    for i, paciente in enumerate(reversed(st.session_state.historial_pacientes[:5])):
        with st.expander(f"Paciente #{len(st.session_state.historial_pacientes) - i} - {paciente['timestamp']}"):
            col_det1, col_det2 = st.columns(2)
            
            with col_det1:
                st.write(f"**Descripción:** {paciente['descripcion']}")
                if paciente.get('edad'):
                    st.write(f"**Edad:** {paciente['edad']} años")
                if paciente.get('genero') and paciente['genero'] != "No especificado":
                    st.write(f"**Género:** {paciente['genero']}")
            
            with col_det2:
                st.write(f"**Temperatura:** {paciente['temperatura']}°C" if paciente['temperatura'] else "**Temperatura:** No especificada")
                st.write(f"**Diagnóstico:** {paciente['diagnostico']}" if paciente['diagnostico'] else "**Diagnóstico:** A determinar")
                st.write(f"**Urgencia:** {paciente['urgencia']}")
            
            if paciente.get("recomendaciones"):
                st.write("**Principales recomendaciones:**")
                for rec in paciente["recomendaciones"][:3]:
                    st.write(f"• {rec}")

# Sidebar con información adicional
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    
    st.write("**Estado actual:**")
    st.success("✅ Análisis de texto activo")
    st.warning("⏳ Reconocimiento de voz en desarrollo")
    
    st.write("**Diagnósticos soportados:**")
    st.write("• Neumonía")
    st.write("• Asma aguda") 
    st.write("• Crisis respiratoria")
    
    st.write("**Códigos NIC incluidos:**")
    st.write("• 3350 - Monitorización Respiratoria")
    st.write("• 3140 - Manejo Vía Aérea")
    st.write("• 6550 - Protección Infecciones")
    st.write("• 5602 - Enseñanza Proceso")
    
    st.markdown("---")
    st.write("**📊 Estadísticas:**")
    if st.session_state.historial_pacientes:
        total = len(st.session_state.historial_pacientes)
        urgencias = [p['urgencia'] for p in st.session_state.historial_pacientes]
        criticas = urgencias.count('critica')
        altas = urgencias.count('alta')
        
        st.write(f"Total evaluaciones: {total}")
        st.write(f"Urgencias críticas: {criticas}")
        st.write(f"Urgencias altas: {altas}")
    else:
        st.write("Sin evaluaciones aún")

# Footer
st.markdown("---")
st.markdown("**⚠️ Importante:** Este es un sistema de apoyo educativo. Siempre confirme diagnósticos con profesionales médicos.")
st.markdown("**🔄 Reconocimiento de voz:** Estará disponible cuando se resuelvan los problemas de compatibilidad de Python en Streamlit Cloud.")
