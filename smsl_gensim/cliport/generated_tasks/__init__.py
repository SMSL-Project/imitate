from importlib import import_module

from smsl_gensim.task_registry import task_class_names


new_names = {}
for module_name, class_name in task_class_names().items():
    module = import_module(f"{__name__}.{module_name}")
    new_names[module_name.replace("_", "-")] = getattr(module, class_name)
