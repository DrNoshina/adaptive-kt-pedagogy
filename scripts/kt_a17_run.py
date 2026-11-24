import runpy, sys, types
try: import shap
except Exception: sys.modules["shap"] = types.ModuleType("shap")
try:
    import matplotlib; matplotlib.use("Agg")
except Exception: pass
sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")
