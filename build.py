def build():
    from subprocess import run
    from shutil import rmtree, copytree, copyfile
    from os.path import exists
    from pathlib import Path
    
    if exists("docs"):
        rmtree("docs")
    result = run(["sphinx-build", ".", "docs"],capture_output=True)
    print(result.stdout.decode("utf-8"))
    if exists("jupyter_execute"):
        rmtree("jupyter_execute")
    Path("docs/.nojekyll").touch()
    
    result = run("jupyter-nbconvert --to slides proposal.ipynb --output docs/proposal --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell' --TagRemovePreprocessor.remove_input_tags='remove-nb-input'",capture_output=True, shell=True)
    result = run("jupyter-nbconvert --to slides present_update.ipynb --output docs/present_update --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell' --TagRemovePreprocessor.remove_input_tags='remove-nb-input'",capture_output=True, shell=True)
    result = run("jupyter-nbconvert --to slides SP.ipynb --output docs/SP --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='remove-nb-cell' --TagRemovePreprocessor.remove_input_tags='remove-nb-input'",capture_output=True, shell=True)

if __name__ == '__main__':
    build()
