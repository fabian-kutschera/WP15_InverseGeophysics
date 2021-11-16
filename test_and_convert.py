import pathlib
import subprocess


ALL_NOTEBOOKS = [
    "00/Line fitting in 1-D.py",
    "01/Bayes' table/Bayes' table.py",
    "01/Find the fisherman/Find the fisherman.py",
    "01/Gauss coefficients/Gauss coefficients.py",
    "01/Gaussian samples/Gaussian samples.py",
    "01/MALA/MALA.py",
    "01/Metropolis-Hastings/Metropolis-Hastings.py",
    "01/Metropolis-Hastings Time shift/Metropolis-Hastings time shift.py",
    "01/Transdimensional sampling/Transdimensional - Naive.py",
    "01/Transdimensional sampling/Transdimensional - Reversible Jump.py",
    "02/1D wave equation/FD1D visco-elastic.py",
    "02/Backus-Gilbert/Love Waves in Layered Media.py",
    "02/Iterative linearisation/Iterative linearisation.py",
    "02/Nonlinear optimisation in 2D/Nonlinear optimisation in 2D.py",
    "02/Straight-ray tomography/Straight-Ray Tomography.py",
    "04/Hamiltonian Nullspace Shuttles/Hamiltonian Nullspace Shuttle - 2D Analytic random.py",
    "04/Hamiltonian Nullspace Shuttles/Hamiltonian Nullspace Shuttle - 2D Analytic.py",
    # "0X/FD2Dpy/FD2Dpy.py"
]


for nb in ALL_NOTEBOOKS:
    print(f"Processing notebook {nb}")
    input_filename = pathlib.Path(nb)
    output_filename = pathlib.Path(
        input_filename.parent,
        input_filename.with_suffix(".ipynb").name.replace(" ", "_").replace("'", ""),
    )
    p = subprocess.run(
        [
            "jupytext",
            "--to",
            "ipynb",
            "-o",
            str(output_filename),
            str(input_filename),
        ],
        capture_output=True,
    )

    if p.returncode != 0:
        print("stdout\n++++++++++++++++++++++", p.stdout)
        print("--------------------------------------------------------------")
        print("stderr\n++++++++++++++++++++++", p.stderr)
        raise ValueError(f"Converting {input_filename} to ipynb failed.")

    p = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=900",
            "--ExecutePreprocessor.store_widget_state=False",
            str(output_filename),
        ],
        capture_output=True,
    )

    if p.returncode != 0:
        print("stdout\n++++++++++++++++++++++", p.stdout)
        print("--------------------------------------------------------------")
        print("stderr\n++++++++++++++++++++++", p.stderr)
        raise ValueError(f"Executing {output_filename} failed.")

    p = subprocess.run(
        [
            "jupyter",
            "nbconvert",
            "--to",
            "markdown",
            "--output-dir=website/docs/notebooks",
            str(output_filename),
        ],
        capture_output=True,
    )

    if p.returncode != 0:
        print("stdout\n++++++++++++++++++++++", p.stdout)
        print("--------------------------------------------------------------")
        print("stderr\n++++++++++++++++++++++", p.stderr)
        raise ValueError(f"Converting {output_filename} to markdown failed.")
