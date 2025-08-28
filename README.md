# 🔥 Wildfire Danger Prediction using Deep Learning + GUI

This project is a **Tkinter-based desktop application** that predicts the probability of wildfire danger from satellite images using a **deep learning model** (TensorFlow/Keras).  
It provides an interactive **login system**, an **image upload option**, and a **visualization of wildfire risk using Class Activation Maps (CAMs)**.

---

## ✨ Features
- 🔑 **Login System** (default credentials: `Admin` / `Admin`)  
- 📂 **Upload Images** (`.jpg`, `.png`, `.jpeg`) for prediction  
- 🤖 **Deep Learning Model** predicts wildfire probability 
- 🌍 **Class Activation Map (CAM)** highlights wildfire-prone areas in the image  
- 🚪 **Logout option** to return to login screen  

---

## 📂 Project Structure
WildfireProject/
│── LoginPage.py # Login screen (Tkinter GUI)
│── MainPro.py # Main application with prediction
│── requirements.txt # Dependencies
│── .gitignore # Ignored files
│── images/ # Backgrounds and icons for UI
│── saved_model/ # Trained model used by the app
│ └── custom_model/

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/lahari-tech25/WildfireProject.git
cd WildfireProject

2. Install dependencies
pip install -r requirements.txt

3. Run the application
python MainPro.py

🧪 Usage Guide

Start the app → Login with:

Username: Admin

Password: Admin

Upload a satellite/fire-related image (.jpg, .png, .jpeg).

Click Predict → The app will show:

Probability of Wildfire vs No Wildfire

A heatmap overlay (CAM) highlighting risky areas.

Click Logout to return to login page.

📦 Dependencies

Python 3.8+
TensorFlow / Keras
NumPy
OpenCV
Matplotlib
Seaborn
Pillow (PIL)
Tkinter (bundled with Python on Windows)

(All dependencies are already included in requirements.txt)
