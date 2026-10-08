@echo off
rem Lance FlySim sous Windows (double-clic). Les options sont transmises au serveur,
rem par exemple :  lancer.bat --flies 3
chcp 65001 >nul
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
    echo FlySim a besoin de "uv", le gestionnaire Python ^(https://docs.astral.sh/uv/^).
    choice /m "Installer uv maintenant avec l'installateur officiel"
    if errorlevel 2 (
        echo Installe uv puis relance ce fichier : https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

echo Premier lancement : installation de Python et des dependances, puis telechargement
echo du cerveau (~135 Mo). Les lancements suivants sont immediats.
echo.
uv run python -m flysim.server %*
pause
