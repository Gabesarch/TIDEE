from abc import ABC
from typing import Optional, Dict, Sequence

from allenact.base_abstractions.sensor import SensorSuite, Sensor

try:
    from allenact.embodiedai.sensors.vision_sensors import (
        DepthSensor,
        IMAGENET_RGB_MEANS,
        IMAGENET_RGB_STDS,
    )
except ImportError:
    raise ImportError("Please update to allenact>=0.4.0.")

from rearrangement.baseline_configs.rearrange_base import RearrangeBaseExperimentConfig
from rearrangement.rearrange.sensors import (
    RGBRearrangeSensor,
    UnshuffledRGBRearrangeSensor,
)
from rearrangement.rearrange.tasks import RearrangeTaskSampler


class OnePhaseRGBBaseExperimentConfig(RearrangeBaseExperimentConfig, ABC):
    @classmethod
    def sensors(cls) -> Sequence[Sensor]:
        cnn_type, pretraining_type = cls.CNN_PREPROCESSOR_TYPE_AND_PRETRAINING
        if pretraining_type.strip().lower() == "clip":
            from allenact_plugins.clip_plugin.clip_preprocessors import (
                ClipResNetPreprocessor,
            )

            mean = ClipResNetPreprocessor.CLIP_RGB_MEANS
            stdev = ClipResNetPreprocessor.CLIP_RGB_STDS
        else:
            mean = IMAGENET_RGB_MEANS
            stdev = IMAGENET_RGB_STDS

        return [
            RGBRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=RearrangeBaseExperimentConfig.EGOCENTRIC_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
            UnshuffledRGBRearrangeSensor(
                height=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                width=RearrangeBaseExperimentConfig.SCREEN_SIZE,
                use_resnet_normalization=True,
                uuid=RearrangeBaseExperimentConfig.UNSHUFFLED_RGB_UUID,
                mean=mean,
                stdev=stdev,
            ),
        ]

    @classmethod
    def make_sampler_fn(
        cls,
        stage: str,
        force_cache_reset: bool,
        allowed_scenes: Optional[Sequence[str]],
        seed: int,
        epochs: int,
        scene_to_allowed_rearrange_inds: Optional[Dict[str, Sequence[int]]] = None,
        x_display: Optional[str] = None,
        sensors: Optional[Sequence[Sensor]] = None,
        thor_controller_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> RearrangeTaskSampler:
        """Return a RearrangeTaskSampler."""
        sensors = cls.sensors() if sensors is None else sensors
        if "mp_ctx" in kwargs:
            del kwargs["mp_ctx"]
        assert not cls.RANDOMIZE_START_ROTATION_DURING_TRAINING
        return RearrangeTaskSampler.from_fixed_dataset(
            run_walkthrough_phase=False,
            run_unshuffle_phase=True,
            stage=stage,
            allowed_scenes=allowed_scenes,
            scene_to_allowed_rearrange_inds=scene_to_allowed_rearrange_inds,
            rearrange_env_kwargs=dict(
                force_cache_reset=force_cache_reset,
                **cls.REARRANGE_ENV_KWARGS,
                controller_kwargs={
                    "x_display": x_display,
                    **cls.THOR_CONTROLLER_KWARGS,
                    **(
                        {} if thor_controller_kwargs is None else thor_controller_kwargs
                    ),
                    "renderDepthImage": any(
                        isinstance(s, DepthSensor) for s in sensors
                    ),
                },
            ),
            seed=seed,
            sensors=SensorSuite(sensors),
            max_steps=cls.MAX_STEPS,
            discrete_actions=cls.actions(),
            require_done_action=cls.REQUIRE_DONE_ACTION,
            force_axis_aligned_start=cls.FORCE_AXIS_ALIGNED_START,
            epochs=epochs,
            **kwargs,
        )
