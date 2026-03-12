import streamlit as st
import requests

st.set_page_config(page_title="VAMA AI Assistant", page_icon="🏢")
st.title("💬 VAMA - Asistente de Ventas")

# 1. Inicializar historial si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Mostrar historial acumulado (Lo que ya se habló)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Input del usuario
user_input = st.chat_input("Escribe aquí...")

if user_input:
    # A. Mostrar mensaje del usuario inmediatamente
    with st.chat_message("user"):
        st.markdown(user_input)
    # Guardar en memoria de sesión
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # B. Llamada al Backend (Flask)
    try:
        with st.spinner("VAMA está pensando..."):
            response = requests.post(
                "http://localhost:5000/webhook",
                json={
                    "message": user_input,
                    "user_id": "5511223344", 
                    "nombre": "Julian Lugo"
                },
                timeout=30 # Evita que se quede colgado eternamente
            )
            full_response = response.json().get("respuesta", "Error: No se recibió respuesta")
    except Exception as e:
        full_response = f"Error de conexión: {str(e)}"

    # C. Mostrar respuesta del bot inmediatamente
    with st.chat_message("assistant"):
        st.markdown(full_response)
    
    # D. Guardar respuesta en memoria de sesión
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Forzar actualización de la UI
    st.rerun()