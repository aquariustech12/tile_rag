import json

with open("conversaciones_completas.jsonl", "r") as f_in, open("dataset.jsonl", "w") as f_out:
    for line in f_in:
        data = json.loads(line)
        user = data["mensaje_usuario"]
        assistant = data["respuesta_bot"]
        if not user or not assistant:
            continue
        # Formato chat de Qwen
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
        f_out.write(json.dumps({"text": prompt}) + "\n")