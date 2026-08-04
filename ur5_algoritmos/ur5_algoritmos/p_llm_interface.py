import requests
import json
import re
import numpy as np

def call_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    return response.json()["response"]

def build_prompt(user_input):
    return f"""
Eres un sistema que convierte instrucciones en parámetros geométricos.

Devuelve SOLO JSON válido.

Formato:
{{"shape":"circle","radius":0.05}}

Entrada: "{user_input}"
Salida:
"""

def extract_json(text):
    matches = re.findall(r'\{.*?\}', text, re.DOTALL)

    for m in matches:
        try:
            return json.loads(m)
        except:
            continue

    raise ValueError("No JSON válido")

def generate_circle(radius, n_points=150):
    t = np.linspace(0, 2*np.pi, n_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    return np.vstack((x, y)).T

def traj2d_to_3d(traj2d, center):
    traj3d = []

    for p in traj2d:
        x = center[0] + p[0]
        y = center[1] + p[1]
        z = center[2]

        pose = np.array([x, y, z, 0, 1, 0, 0])
        traj3d.append(pose)

    return traj3d

def get_trajectory_3d(user_input):

    prompt = build_prompt(user_input)
    raw = call_llm(prompt)
    parsed = extract_json(raw)

    if parsed["shape"] == "circle":
        r = np.clip(float(parsed.get("radius", 0.05)), 0.01, 0.1)
        traj2d = generate_circle(r)
    else:
        raise ValueError("Figura no soportada")

    center = np.array([0.4, 0.24, 0.1])
    return traj2d_to_3d(traj2d, center)