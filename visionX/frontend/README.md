# Terminal 1 - Backend (C++ engine)
cd backend
uvicorn main:app --port 8000 --reload

# Terminal 2 - Frontend React
cd frontend
npm install
npm run dev
# Open http://localhost:5173