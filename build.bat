@echo off
cd frontend
call npm install
call npm run build
cd ..
if exist backend\static rmdir /s /q backend\static
xcopy /e /i frontend\dist backend\static
echo Build complete. Run: cd backend ^&^& uv run python run.py
