Image Location Finder using ML and Flask

This project is a web-based machine learning application that allows users to upload an image and get the predicted location shown in the image — including country, city, and landmark — using deep learning and geolocation models.


✨ Features

🖼 Upload any place image (e.g., landmark)

🧠 Predicts country, city, and optionally landmark

📍 Shows a link to view the predicted location on Google Maps

🎨 Clean responsive UI built with Tailwind CSS

🔁 Model powered by CLIP or Image2GPS/StreetCLIP

🌐 Fully deployable on platforms like Render or Railway


🛠 Tech Stack

Python + Flask (backend)

HTML + Tailwind CSS (frontend)

PyTorch / Transformers (ML inference)

StreetCLIP or CLIP for image embedding

Custom hierarchical prediction logic (Country → City → Landmark)

