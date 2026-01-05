project('tdscha',
  ['c'],
  version: '1.5.1',
  license: 'GPL',
  meson_version: '>= 1.1.0', # <- set min version of meson.
  default_options : [
        'warning_level=1',
        'buildtype=release',
        'c_args=-O2'
    ]
    )

# --- Compilation options ---
use_mkl = get_option('use_mkl')

# --- System and Python Dependencies ---
# Find the necessary installations
py = import('python').find_installation(pure: false)
py_dep = py.dependency()
cc = meson.get_compiler('c')

# Additional dependencies

# The LAPACK and BLAS libraries are essential for Fortran matrix operations.
#lapack_dep = dependency('lapack')
#blas_dep = dependency('blas')
openblas_dep = dependency('openblas', required: false)
if use_mkl
  mkl_dep = dependency('mkl', required: true)
  blas_dep = mkl_dep
  lapack_dep = mkl_dep
else
  lapack_dep = dependency('lapack', required: true)
  blas_dep = dependency('blas', required: true)
endif

# The openmp dependency is necessary
openmp_dep = dependency('openmp', required: true)

# --- MPI Detection ---
# This overrides the logic in os.environ["MPICC"] and os.popen("%s -show" % mpicc).
# Meson has a built-in MPI module.
mpi_args = []
mpi_link_args = []
mpi_compile_args = []
has_mpi = false

# Attempts to find the MPI dependency.
# You can specify a specific MPI compiler with the 'mpi_compiler' parameter
# or, if you want, a specific Fortran compiler with 'mpi_fortran_compiler'.
# For OpenMPI, IntelMPI, MPICH, etc., Meson usually finds it automatically.
#mpi_dep = dependency('mpi', required: false, language: ['c', 'fortran'])
mpi_dep = dependency('mpi', required: false)

if mpi_dep.found()
  message('MPI environment detected correctly.')
  has_mpi = true
  # Meson handles adding appropriate flags. We just add the define.
  # If you need specific MPI flags beyond what Meson adds automatically,
  # you can get them via mpi_dep.get_compile_args() and mpi_dep.get_link_args()
  # and add them to extra_compile_args/extra_link_args.
  mpi_compile_args += ['-D_MPI']
else
  # Here you can add warning logic if MPI is not found.
  # Meson prints a warning if required: true and it is not found.
  # For required: false, you can print your own warning.
  warning('No MPI compiler found, please ensure MPI is installed and configured.')
  warning('If you wish to activate MPI acceleration, consider setting MPICC environment variable or providing Meson with appropriate flags.')
endif

# --- NUMPY CONFIGURATION ---
# Gets the path to the NumPy and f2py header directories using the Python command
incdir_numpy = run_command(py,
    ['-c', 'import numpy; print(numpy.get_include())'],
    check : true
).stdout().strip()

# f2py also requires the fortranobject.h header
incdir_f2py = run_command(py,
    ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)
np_dep = declare_dependency(include_directories: inc_np)

# --- END OF NUMPY CONFIGURATION ---
if openblas_dep.found()
  message('openblas environment detected correctly.')
  list_dep = [py_dep, mpi_dep, openblas_dep, lapack_dep, blas_dep]
else
  warning('No openblas found.')
  list_dep = [py_dep, mpi_dep, lapack_dep, blas_dep]
endif
# --- Definition of each Python extension (Fortran) ---

# 'symph' extension

# Compilation of the Fortran module: symph
sources_cfiles = ['CModules/odd_corr_module.c', 'CModules/LanczosFunctions.c']

py.extension_module('sscha_HP_odd',
    sources_cfiles,
    include_directories: inc_np,
    dependencies: list_dep,
#    link_args: ['-L' + py.get_install_dir() / 'numpy' / 'f2py' / 'src' / 'fortranobject.c', '-lfortranobject'],
    install: true
)

# --- Installation of the 'cellconstructor' Python package ---
#install_data(
#  'cellconstructor/__init__.py', 'cellconstructor/AnharmonicForceFields.py', 'cellconstructor/calculators.py',
#  'cellconstructor/Methods.py', 'cellconstructor/Phonons.py', 'cellconstructor/Spectral.py',
#  'cellconstructor/ThermalConductivity.py', 'cellconstructor/Units.py', 'cellconstructor/Bands.py',
#  'cellconstructor/ForceTensor.py', 'cellconstructor/Manipulate.py', 'cellconstructor/Moro_object.py',
#  'cellconstructor/Settings.py', 'cellconstructor/Structure.py', 'cellconstructor/symmetries.py',
#  'cellconstructor/Timer.py',
#  install_dir: py.get_install_dir() / 'cellconstructor',
#)
py.install_sources(
    [
      'Modules/__init__.py',
      'Modules/DynamicalLanczos.py',
      'Modules/Dynamical.py',
      'Modules/Parallel.py',
      'Modules/Perturbations.py',
      'Modules/StaticHessian.py',
      'Modules/tdscha_core.jl',
      'Modules/Tools.py',
      'Modules/cli.py'
    ],
    subdir: 'tdscha'
)

install_data(
  'Modules/tdscha_core.jl',
  install_dir: py.get_install_dir() / 'Modules'
)

# --- Installation of executable scripts ---
py.install_sources([
  'scripts/plot_hessian_convergence.py',
  'scripts/tdscha-convergence-analysis.py',
  'scripts/tdscha-output2abc.py',
  'scripts/tdscha-plot.py'
])

# Set the tests by pytest.
pytest_exe = find_program('pytest', required: false)

if pytest_exe.found()
  test('pytest', pytest_exe,
    args : ['-v'],
    workdir : meson.project_source_root()
  )
else
  message('pytest not found; pytest tests are skipped.')
endif
