pyinstaller main.py --noconfirm ^
    --add-data="package.json;." ^
    --add-data="package-lock.json;." ^
    --add-data="electron;electron" ^
    --add-data="node_modules;node_modules"