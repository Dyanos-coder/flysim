#!/usr/bin/env sh
# Lance FlySim sous macOS / Linux. Les options sont transmises au serveur,
# par exemple :  ./lancer.sh --flies 3
set -e
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
    echo 'FlySim a besoin de "uv", le gestionnaire Python (https://docs.astral.sh/uv/).'
    printf "Installer uv maintenant avec l'installateur officiel ? [o/N] "
    read -r answer
    case "$answer" in
        o|O|oui|y|Y) curl -LsSf https://astral.sh/uv/install.sh | sh ;;
        *) echo "Installe uv puis relance : https://docs.astral.sh/uv/getting-started/installation/"; exit 1 ;;
    esac
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Premier lancement : installation de Python et des dépendances, puis téléchargement"
echo "du cerveau (~135 Mo). Les lancements suivants sont immédiats."
echo
exec uv run python -m flysim.server "$@"
