from setuptools import setup, Extension
from torch.utils import cpp_extension

__version__ = "0.2.3"

setup(name='dct_manip',
      ext_modules=[cpp_extension.CppExtension(
	      'dct_manip', 
	      ['dct_manip.cpp'],
		  include_dirs=['/home/jespark/mambaforge/envs/rgbnomore/include'],
		  library_dirs=['/home/jespark/mambaforge/envs/rgbnomore/lib'],
	      extra_objects=[
			'/home/jespark/mambaforge/envs/rgbnomore/lib/libjpeg.so',
			],
		  headers=[
			'/home/jespark/mambaforge/envs/rgbnomore/include/jpeglib.h',],
	      extra_compile_args=['-std=c++17']
	      ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
	  version = __version__
	  )
