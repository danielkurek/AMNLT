from dataclasses import dataclass
from typing import Any, TypeVar, Type, cast


T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x

def from_optional_int(x: Any) -> int | None:
    if isinstance(x, str) and len(x) == 0:
        return None
    assert isinstance(x, int) or x is None
    return x

def from_float(x: Any) -> float:
    assert isinstance(x, (float, int)) and not isinstance(x, bool)
    return float(x)

def from_bool(x: Any) -> bool:
    assert isinstance(x, (bool, int))
    return bool(x)


def to_float(x: Any) -> float:
    assert isinstance(x, (int, float))
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


@dataclass
class Data:
    dataset_name: str
    reduce_ratio: float
    transcript_format: str
    use_for_validation: bool
    downsample: bool

    @staticmethod
    def from_dict(obj: Any) -> 'Data':
        assert isinstance(obj, dict)
        dataset_name = from_str(obj.get("dataset_name"))
        reduce_ratio = from_float(obj.get("reduce_ratio"))
        transcript_format = from_str(obj.get("transcript_format"))
        use_for_validation = from_bool(obj.get("use_for_validation", True))
        downsample = from_bool(obj.get("downsample", False))
        return Data(dataset_name, reduce_ratio, transcript_format, use_for_validation, downsample)

    def to_dict(self) -> dict:
        result: dict = {}
        result["dataset_name"] = from_str(self.dataset_name)
        result["reduce_ratio"] = to_float(self.reduce_ratio)
        result["transcript_format"] = from_str(self.transcript_format)
        result["use_for_validation"] = from_bool(self.use_for_validation)
        result["downsample"] = from_bool(self.downsample)
        return result


@dataclass
class ExperimentCommonConfig:
    name: str
    vocab_name: str
    batch_size: int
    num_workers: int
    downsample_size: None | int
    data: list[Data]

    @staticmethod
    def from_dict(obj: Any) -> 'ExperimentCommonConfig':
        assert isinstance(obj, dict)
        name = from_str(obj.get("name"))
        vocab_name = from_str(obj.get("vocab_name"))
        batch_size = from_int(obj.get("batch_size"))
        num_workers = from_int(obj.get("num_workers"))
        downsample_size = obj.get("downsample_size")
        data = [Data.from_dict(data_dict) for data_dict in obj.get("data")]
        return ExperimentCommonConfig(name, vocab_name, batch_size, num_workers, downsample_size, data)

    def to_dict(self) -> dict:
        result: dict = {}
        result["data"] = to_class(Data, self.data)
        return result


def experiment_config_from_dict(s: Any) -> ExperimentCommonConfig:
    return ExperimentCommonConfig.from_dict(s)


def experiment_config_to_dict(x: ExperimentCommonConfig) -> Any:
    return to_class(ExperimentCommonConfig, x)