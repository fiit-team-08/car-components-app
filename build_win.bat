pyinstaller main.py --noconfirm ^
    --noconsole ^
    --add-data="package.json;." ^
    --add-data="package-lock.json;." ^
    --add-data="electron;electron" ^
    --add-data="node_modules;node_modules"