from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from query_tiles_rag import buscar_vendedor
import uvicorn

app = FastAPI(title="TileRAG_API")

# Definimos qué datos esperamos recibir (JSON)
class Consulta(BaseModel):
    pregunta: str
    metros: float = None

@app.get("/")
def check():
    return {"status": "IA Server Online", "gpu": "RTX 3090 Ready"}

@app.post("/consultar")
def api_consultar(data: Consulta):
    try:
        # Ejecutamos tu lógica de búsqueda y cálculo
        resultado = buscar_vendedor(data.pregunta, data.metros)
        return {"respuesta": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Escuchamos en el puerto 8000
    uvicorn.run(app, host="0.0.0.0", port=8005)