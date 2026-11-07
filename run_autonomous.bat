@echo off
echo =======================================================
echo   STARTING PIT WALL AUTONOMOUS DRIVER (SpatialResNet18)
echo =======================================================
echo.
start "" "simulator\Default Windows desktop 64-bit.exe"
.venv\Scripts\python -m src.drive --model models/resnet18_pro.pt --smoothing 0.20 --steer-gain 1.25 --max-speed 20.0
pause
