"""This module defines the Config class.
This class is responsible for managing the options selected for the user and
contains the default configuration.
"""
from importlib import metadata
from configparser import ConfigParser
import logging
import multiprocessing
import os
import re
from datetime import datetime
import git
from git.exc import InvalidGitRepositoryError

from yt.utilities.logger import set_log_level as set_log_level_yt
from yt.config import ytcfg

from sbla_in_sim._version import __version__ as sbla_in_sim_version
from sbla_in_sim.errors import ConfigError
from sbla_in_sim.utils import setup_logger, PROGRESS_LEVEL_NUM

try:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    SBLA_IN_DIR_BASE = THIS_DIR.split("sbla_in_sim/")[0]
    git_hash = git.Repo(SBLA_IN_DIR_BASE).head.object.hexsha
except InvalidGitRepositoryError:  # pragma: no cover
    git_hash = metadata.metadata('sbla_in_sim')['Summary'].split(':')[-1]

ACCEPTED_LOGGING_OPTIONS = [
    "log file", "logging level console", "logging level file", "log level yt"
]
ACCEPTED_GENERAL_OPTIONS = [
    "continue previous run", "out dir", "output catalogue", "overwrite", "num processors"
]

ACCEPTED_RANDOM_RAYS_OPTIONS = [
    "noise dist", "num rays", "random seed", 
    "ray base name", "snapshots", "snapshots dir", "z_dist",
]

DEFAULT_CONFIG = {
    "general": {
        "continue previous run": True,
        "num processors": 0,
        "overwrite": True,
    },
    "logging": {
        "log file": "run.log",
        # New logging level defined in setup_logger.
        # Numeric value is PROGRESS_LEVEL_NUM defined in utils.py
        "logging level console": "PROGRESS",
        "logging level file": "PROGRESS",
        "log level yt": "ERROR",  # yt log level  
    },
    "random rays": {
        "num rays": 1000,
        "random seed": 45737353,
        "rays base name": "ray_",
    },
    "run specs": {
        "git hash": git_hash,
        "timestamp": str(datetime.now()),
        "version": sbla_in_sim_version,
    },
    
}


class Config:
    """Class to manage the configuration file

    Methods
    -------
    __init__
    __format_general_section
    __format_logging_section
    __parse_environ_variables
    initialize_folders
    write_config

    Attributes
    ---------
    config: ConfigParser
    A ConfigParser instance with the user configuration

    continue_previous_run: bool
    If True, continue a previous run. If False, start a new run

    log: str or None
    Name of the log file. None for no log file

    logger: logging.Logger
    Logger object

    logging_level_console: str
    Level of console logging. Messages with lower priorities will not be logged.
    Accepted values are (in order of priority) NOTSET, DEBUG, PROGRESS, INFO,
    WARNING, WARNING_OK, ERROR, CRITICAL.

    logging_level_file: str
    Level of file logging. Messages with lower priorities will not be logged.
    Accepted values are (in order of priority) NOTSET, DEBUG, PROGRESS, INFO,
    WARNING, WARNING_OK, ERROR, CRITICAL.

    log_level_yt: str
    Level of yt logging. Messages with lower priorities will not be logged.
    Accepted values are (in order of priority) NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL.

    num_processors: int
    Number of processors to use for multiprocessing-enabled tasks (will be passed
    downstream to relevant classes like e.g. ExpectedFlux or Data)

    num_rays: int
    Number of random rays to extract

    out_dir: str
    Name of the directory where the deltas will be saved

    overwrite: bool
    If True, overwrite a previous run in the saved in the same output
    directory. Does not have any effect if the folder `out_dir` does not
    exist.

    random_seed: int
    Random seed for the random number generator. This is used to ensure
    reproducibility of the results. 

    rays_base_name: str
    Base name for the random rays. The full name will be
    `rays_base_name` + `snapshot` + `_z` + `redshift` + `_x` + `x_start` + 
    `_x_end` + `_y` + `y_start` + `_y_end` + `_z` + `z_start` + `_z_end`.

    snapshots: str
    Name of the file containing the list of snapshots to extract the rays from.
    The file should contain one snapshot per line. Each line should contain
    the name of the snapshot, the redshift, and the position of the galaxy in
    the form `x y z`. The format of each line is:
    name rho_max z_max galaxy_pos_x  galaxy_pos_y  galaxy_pos_z

    snapshots_dir: str
    Directory where the snapshots are stored. This is used to load the
    snapshots when extracting the rays.
    """

    def __init__(self, filename):
        """Initializes class instance

        Arguments
        ---------
        filename: str
        Name of the config file
        """
        self.logger = logging.getLogger(__name__)

        self.config = ConfigParser()
        # with this we allow options to use capital letters
        self.config.optionxform = lambda option: option
        # load default configuration
        self.config.read_dict(DEFAULT_CONFIG)
        # now read the configuration file
        if os.path.isfile(filename):
            self.config.read(filename)
        else:
            raise ConfigError(f"Config file not found: {filename}")

        # parse the environ variables
        self.__parse_environ_variables()

        # format the sections
        self.continue_previous_run = None
        self.overwrite = None
        self.num_processors = None
        self.out_dir = None
        self.output_catalogue = None
        self.__format_general_section()

        self.log = None
        self.logging_level_console = None
        self.logging_level_file = None
        self.log_level_yt = None
        self.__format_logging_section()

        self.noise_dist = None
        self.num_rays = None
        self.random_seed = None
        self.rays_base_name = None
        self.snapshots = None
        self.snapshots_dir = None
        self.z_dist = None
        self.__format_random_rays_section()

        # initialize folders where data will be saved
        self.initialize_folders()

        # setup logger
        setup_logger(logging_level_console=self.logging_level_console,
                     log_file=self.log,
                     logging_level_file=self.logging_level_file)
        # set yt log level
        set_log_level_yt(self.log_level_yt)
        ytcfg.update({"yt": {"suppress_stream_logging": True}})


    def __format_general_section(self):
        """Format the general section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        # this should never be true as the general section is loaded in the
        # default dictionary
        if "general" not in self.config:  # pragma: no cover
            raise ConfigError("Missing section [general]")
        section = self.config["general"]

        # check that arguments are valid
        for key in section.keys():
            if key not in ACCEPTED_GENERAL_OPTIONS:
                raise ConfigError("Unrecognised option in section [general]. "
                                  f"Found: '{key}'. Accepted options are "
                                  f"{ACCEPTED_GENERAL_OPTIONS}")

        self.continue_previous_run = section.getboolean("continue previous run")
        if self.continue_previous_run is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'continue previous run' in section [general]")
        
        self.num_processors = section.getint("num processors")
        # this should never be true as the general section is loaded in the
        # default dictionary
        if self.num_processors is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'num processors' in section [general]")
        if self.num_processors == 0:
            self.num_processors = (multiprocessing.cpu_count() // 2)
    
        self.out_dir = section.get("out dir")
        if self.out_dir is None:
            raise ConfigError("Missing variable 'out dir' in section [general]")
        if not self.out_dir.endswith("/"):
            self.out_dir += "/"

        self.output_catalogue = section.get("output catalogue")
        if self.output_catalogue is None:
            raise ConfigError(
                "Missing variable 'output catalogue' in section [general]")
        
        self.overwrite = section.getboolean("overwrite")
        # this should never be true as the general section is loaded in the
        # default dictionary
        if self.overwrite is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'overwrite' in section [general]")
        
    def __format_logging_section(self):
        # this should never be true as the logging section is loaded in the
        # default dictionary
        if "logging" not in self.config:  # pragma: no cover
            raise ConfigError("Missing section [logging]")
        section = self.config["logging"]

        self.log = section.get("log file")
        # this should never be true as the general section is loaded in the
        # default dictionary
        if self.log is None:  # pragma: no cover
            raise ConfigError("Missing variable 'log file' in section [logging]")
        if "/" in self.log:
            raise ConfigError(
                "Variable 'log file' in section [logging] should not incude folders. "
                f"Found: {self.log}")
        self.log = self.out_dir + "Log/" + self.log
        section["log file"] = self.log

        self.logging_level_console = section.get("logging level console")
        # this should never be true as the logging section is loaded in the
        # default dictionary
        if self.logging_level_console is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'logging level console' in section [logging]")
        self.logging_level_console = self.logging_level_console.upper()

        self.logging_level_file = section.get("logging level file")
        # this should never be true as the logging section is loaded in the
        # default dictionary
        if self.logging_level_file is None:  # pragma: no cover
            raise ConfigError(
                "In section 'logging level file' in section [logging]")
        self.logging_level_file = self.logging_level_file.upper()

        self.log_level_yt = section.get("log level yt")
        # this should never be true as the logging section is loaded in the
        # default dictionary
        if self.log_level_yt is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'log level yt' in section [logging]")
        self.log_level_yt = self.log_level_yt.upper()

    def __format_random_rays_section(self):
        """Format the random rays section of the parser into usable data

        Raise
        -----
        ConfigError if the config file is not correct
        """
        # this should never be true as the random rays section is loaded in the
        # default dictionary
        if "random rays" not in self.config:  # pragma: no cover
            raise ConfigError("Missing section [random rays]")
        section = self.config["random rays"] 

        self.num_rays = section.getint("num rays")
        # this should never be true as the logging section is loaded in the
        # default dictionary
        if self.num_rays is None: # pragma: no cover
            raise ConfigError("Missing variable 'num rays' in section [random rays]")

        self.snapshots = section.get("snapshots")
        if self.snapshots is None:
            raise ConfigError("Missing variable 'snapshots' in section [random rays]")

        self.snapshots_dir = section.get("snapshots dir")
        if self.snapshots_dir is None:
            raise ConfigError("Missing variable 'snapshots dir' in section [random rays]")

        self.z_dist = section.get("z_dist")
        if self.z_dist is None:
            raise ConfigError("Missing variable 'z_dist' in section [random rays]")

        self.random_seed = section.getint("random_seed")
        # this should never be true as the general section is loaded in the
        # default dictionary
        if self.random_seed is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'random_seed' in section [random rays]")

        self.rays_base_name = section.get("rays base name")
        # this should never be true as the general section is loaded in the
        # default dictionary
        if self.rays_base_name is None:  # pragma: no cover
            raise ConfigError(
                "Missing variable 'rays base name' in section [random rays]")

    def __parse_environ_variables(self):
        """Read all variables and replaces the enviroment variables for their
        actual values. This assumes that enviroment variables are only used
        at the beggining of the paths.

        Raise
        -----
        ConfigError if an environ variable was not defined
        """
        for section in self.config:
            for key, value in self.config[section].items():
                if value.startswith("$"):
                    pos = value.find("/")
                    if os.getenv(value[1:pos]) is None:
                        raise ConfigError(
                            f"In section [{section}], undefined "
                            f"environment variable {value[1:pos]} "
                            "was found")
                    self.config[section][key] = value.replace(
                        value[:pos], os.getenv(value[1:pos]))

    def initialize_folders(self):
        """Initialize output folders

        Raise
        -----
        ConfigError if the output path was already used and the
        overwrite is not selected
        """
        if not os.path.exists(f"{self.out_dir}/config.ini"):
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(self.out_dir + "Delta/", exist_ok=True)
            os.makedirs(self.out_dir + "Log/", exist_ok=True)
            self.write_config()
        elif self.overwrite:
            os.makedirs(self.out_dir + "Delta/", exist_ok=True)
            os.makedirs(self.out_dir + "Log/", exist_ok=True)
            self.write_config()
        else:
            raise ConfigError("Specified folder contains a previous run. "
                              "Pass overwrite option in configuration file "
                              "in order to ignore the previous run or "
                              "change the output path variable to point "
                              f"elsewhere. Folder: {self.out_dir}")

    def write_config(self):
        """This function writes the configuration options for later
        usages. The file is saved under the name .config.ini and in
        the self.out_dir folder
        """
        outname = f"{self.out_dir}/.config.ini"
        if os.path.exists(outname):
            newname = f"{outname}.{os.path.getmtime(outname)}"
            os.rename(outname, newname)
        with open(outname, 'w', encoding="utf-8") as config_file:
            self.config.write(config_file)
