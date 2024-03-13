#This file contains the start of wrapping the model into a context manager - not yet used or complete

from __future__ import annotations
from optparse import Option

from rich.progress import Progress, TextColumn, TaskID
from concurrent.futures import ThreadPoolExecutor, Future
from copy import copy
import tomli
import pandas as pd
from .modeling import Model
from .config import construct_case_study
from typing import Optional
from supermodel import CONFIG_PATH

if CONFIG_PATH is not None:
    with open(CONFIG_PATH, 'rb') as setup_file:
        setup_params = tomli.load(setup_file)
else:
    raise Exception('CONFIG_PATH not set, please run supermodel.loadconfig(path)')

class job:

    def __init__(self, model, repeats=1, to_csv: bool = False, output_path = setup_params['environment']['output_path'], other_info: Optional[str] = ''):
        """Class to hold job data that is submitted to the context manager to be run concurrently. Holds the model, number of repeats, and other info.
        model: Model object to run
        repeats: number of times to run the model
        to_csv: if True, output data to csv
        output_path: path to output csvs to, defaults to setup.toml value
        other_info: string to append to output csv name
        """
        self.model: Model = model
        self.task_id: Optional[TaskID] = None
        self.repeats: int = repeats
        self.to_csv: bool = to_csv
        self.output_path: str = output_path
        self.other_info: Optional[str] = other_info
        self.results: dict[int, Future] = {} #dict to hold results

class supermodel_context:
    def __init__(self, max_threads=4):
        """Context manager to run multiple models in parallel. Runs the model concurrently with the specified number of repeats, returns list of the output dataframes, for the user to do as they please with."""
        self.jobs: list[job] = [] #list to hold jobs 
        self._max_threads = max_threads
        self._num_jobs = 0

    @property
    def num_jobs(self):
        return len(self.jobs)

    def __enter__(self) -> supermodel_context:
        self.progress = Progress(
        *Progress.get_default_columns(),
        TextColumn("[red]# infected: [bold]{task.fields[num_infected]}"),
        TextColumn("[purple]viral load: [bold]{task.fields[viral_load]}"),
        TextColumn("[cyan]model time: [bold]{task.fields[time]}"),
        ).__enter__()

        self.pool = ThreadPoolExecutor(max_workers=self._max_threads).__enter__()
        try:
            return self
        except Exception as e:
            self.pool.__exit__(exc_type=type(e), exc_val=e, exc_tb=e.__traceback__)
            self.progress.__exit__(exc_type=type(e), exc_val=e, exc_tb=e.__traceback__)
            raise e
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.pool.__exit__(exc_type, exc_value, traceback)
        self.progress.__exit__(exc_type, exc_value, traceback)

    def add_job(self, 
                model, 
                repeats=1, 
                to_csv: bool = False, 
                output_path = setup_params['environment']['output_path'], 
                other_info: Optional[str] = None
                ):
        """Add a job to the context manager to be run concurrently.
        model: Model object to run
        repeats: number of times to run the model
        to_csv: if True, output data to csv
        output_path: path to output csvs to, defaults to setup.toml value
        other_info: string to append to output csv name"""

        _job = job(
            model=model,
            repeats=repeats,
            to_csv=to_csv,
            output_path=output_path,
            other_info=other_info
        )
        self.progress.print("[bold]Added model: [/bold]" , model)
        _job.results = {}
        for repeat_no in range(repeats):
            future = self.pool.submit(self.__run_job, _job, repeat_no, self.progress)
            _job.results[repeat_no] = future
        self.jobs.append(_job)

        return _job

    def __run_job(self,_job: job, repeat: int,  progress:Progress) -> pd.DataFrame:
        
        task_id = self.progress.add_task("{}: job {} of {}".format(_job.task_id, self.num_jobs),start=False, num_infected=0, viral_load=0, time=0)
        self.progress.start_task(task_id)
        _model: Model = copy.deepcopy(_job.model)
        self.progress.update(task_id, total=_model.max_time)
        _model.run(progress, task_id)
        return _model.output_data
    

    def add_case_study(self, name: str, repeats:int, to_csv: bool = False, output_path = setup_params['environment']['output_path']):
        """Add a case study to the context manager to be run concurrently. Holds the model, number of repeats, and other info.
        name: name of case study to run
        repeats: number of times to run the model
        to_csv: if True, output data to csv
        output_path: path to output csvs to, defaults to setup.toml value
        other_info: string to append to output csv name"""
        _model: Model = construct_case_study(name)
        self.add_job(_model, repeats, to_csv, output_path)