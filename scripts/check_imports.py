import importlib
mods = ['src.config','src.document_processor','src.vector_engine','src.bot_logic']
for m in mods:
    try:
        importlib.import_module(m)
        print('OK', m)
    except Exception as e:
        print('ERR', m, type(e).__name__, e)
