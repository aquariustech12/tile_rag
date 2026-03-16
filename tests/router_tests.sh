#!/usr/bin/env bash
set -e
echo "1) CMX01"
curl -s -X POST "http://127.0.0.1:5000/webhook" -H "Content-Type: application/json" -d '{"mensaje":"CMX01"}' | jq .
echo "2) monomando para lavabo"
curl -s -X POST "http://127.0.0.1:5000/webhook" -H "Content-Type: application/json" -d '{"mensaje":"monomando para lavabo"}' | jq .
echo "3) Receta: huevos estrellados"
curl -s -X POST "http://127.0.0.1:5000/webhook" -H "Content-Type: application/json" -d '{"mensaje":"Receta: huevos estrellados"}' | jq .
echo "4) ¿Dónde está el local?"
curl -s -X POST "http://127.0.0.1:5000/webhook" -H "Content-Type: application/json" -d '{"mensaje":"¿Dónde está el local?"}' | jq .
