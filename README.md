# SeeFood 🌭 vs 🚫🌭

A beginner-friendly Hotdog vs Not Hotdog image classification application powered by FastAPI, TensorFlow / MobileNetV2, and React.

---

## Model Setup Note

Due to GitHub's file size limits and best practices for Git version control, the trained binary model (`seefood_best.keras`) is excluded from Git tracking via `.gitignore`.

Before starting the backend or running inference:
1. Ensure the `models/` directory exists in the project root.
2. Place the trained model file at:
   ```
   models/seefood_best.keras
   ```
The backend service will automatically resolve and load `models/seefood_best.keras` upon startup.
