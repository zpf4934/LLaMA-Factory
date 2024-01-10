import inspect

import gradio as gr
from typing import TYPE_CHECKING, Dict
from transformers.trainer_utils import SchedulerType

from llmtuner.extras.constants import TRAINING_STAGES, REPORT, LOG_LEVEL
from llmtuner.webui.common import list_adapters, list_dataset, get_scripts, DEFAULT_DATA_DIR
from llmtuner.webui.components.data import create_preview_box
from llmtuner.webui.utils import gen_plot

if TYPE_CHECKING:
    from gradio.components import Component
    from llmtuner.webui.engine import Engine


def output_postprocess(y: str | None) -> str | None:
    """
    Parameters:
        y: markdown representation
    Returns:
        HTML rendering of markdown
    """
    if y is None:
        return None
    unindented_y = inspect.cleandoc(y)
    if not unindented_y.startswith("```bash"):
        unindented_y = '\n'.join(unindented_y.split('\n')[:-50:-1])
    return unindented_y

def create_train_tab(engine: "Engine") -> Dict[str, "Component"]:
    input_elems = engine.manager.get_base_elems()
    elem_dict = dict()

    with gr.Row():
        training_stage = gr.Dropdown(
            choices=list(TRAINING_STAGES.keys()), value=list(TRAINING_STAGES.keys())[0], scale=2
        )
        dataset_dir = gr.Textbox(value=DEFAULT_DATA_DIR, scale=2)
        dataset = gr.Dropdown(multiselect=True, scale=4)
        preview_elems = create_preview_box(dataset_dir, dataset)

    training_stage.change(list_dataset, [dataset_dir, training_stage], [dataset], queue=False)
    dataset_dir.change(list_dataset, [dataset_dir, training_stage], [dataset], queue=False)

    input_elems.update({training_stage, dataset_dir, dataset})
    elem_dict.update(dict(
        training_stage=training_stage, dataset_dir=dataset_dir, dataset=dataset, **preview_elems
    ))

    with gr.Row():
        cutoff_len = gr.Slider(value=1024, minimum=4, maximum=32000, step=100)
        learning_rate = gr.Textbox(value="5e-5")
        num_train_epochs = gr.Textbox(value="3.0")
        max_samples = gr.Textbox(value="-1")
        compute_type = gr.Radio(choices=["fp16", "bf16"], value="fp16")

    input_elems.update({cutoff_len, learning_rate, num_train_epochs, max_samples, compute_type})
    elem_dict.update(dict(
        cutoff_len=cutoff_len, learning_rate=learning_rate, num_train_epochs=num_train_epochs,
        max_samples=max_samples, compute_type=compute_type
    ))

    with gr.Row():
        batch_size = gr.Slider(value=4, minimum=1, maximum=512, step=1)
        gradient_accumulation_steps = gr.Slider(value=4, minimum=1, maximum=512, step=1)
        lr_scheduler_type = gr.Dropdown(
            choices=[scheduler.value for scheduler in SchedulerType], value="cosine"
        )
        max_grad_norm = gr.Textbox(value="1.0")
        val_size = gr.Slider(value=0, minimum=0, maximum=1, step=0.001)

    input_elems.update({batch_size, gradient_accumulation_steps, lr_scheduler_type, max_grad_norm, val_size})
    elem_dict.update(dict(
        batch_size=batch_size, gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler_type=lr_scheduler_type, max_grad_norm=max_grad_norm, val_size=val_size
    ))

    with gr.Accordion(label="Extra config", open=False) as extra_tab:
        with gr.Row():
            logging_steps = gr.Slider(value=5, minimum=5, maximum=1000, step=5)
            save_steps = gr.Slider(value=100, minimum=10, maximum=5000, step=10)
            warmup_steps = gr.Slider(value=0, minimum=0, maximum=5000, step=1)
            neftune_alpha = gr.Slider(value=0, minimum=0, maximum=10, step=0.1)

            with gr.Column():
                train_on_prompt = gr.Checkbox(value=False)
                upcast_layernorm = gr.Checkbox(value=False)

        with gr.Row():
            preprocessing_num_workers = gr.Slider(value=5, minimum=1, maximum=30, step=1)
            overwrite_output_dir = gr.Checkbox(value=True, interactive=True)
            overwrite_cache = gr.Checkbox(value=False, interactive=True)
            streaming = gr.Checkbox(value=False, interactive=True)
            report_to = gr.Dropdown(choices=REPORT, value=0, scale=1)
            log_level = gr.Dropdown(choices=LOG_LEVEL, value=0, scale=1)


    input_elems.update({logging_steps, save_steps, warmup_steps, neftune_alpha, train_on_prompt, upcast_layernorm,
                        preprocessing_num_workers, overwrite_output_dir, report_to, overwrite_cache, streaming,
                        log_level})
    elem_dict.update(dict(
        extra_tab=extra_tab, logging_steps=logging_steps, save_steps=save_steps, warmup_steps=warmup_steps,
        neftune_alpha=neftune_alpha, train_on_prompt=train_on_prompt, upcast_layernorm=upcast_layernorm,
        preprocessing_num_workers=preprocessing_num_workers, overwrite_output_dir=overwrite_output_dir,
        report_to=report_to, overwrite_cache=overwrite_cache, streaming=streaming,log_level=log_level
    ))

    with gr.Accordion(label="LoRA config", open=False) as lora_tab:
        with gr.Row():
            lora_rank = gr.Slider(value=8, minimum=1, maximum=1024, step=1, scale=1)
            lora_dropout = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=1)
            lora_target = gr.Textbox(scale=1)
            additional_target = gr.Textbox(scale=1)
            create_new_adapter = gr.Checkbox(scale=1)

    input_elems.update({lora_rank, lora_dropout, lora_target, additional_target, create_new_adapter})
    elem_dict.update(dict(
        lora_tab=lora_tab, lora_rank=lora_rank, lora_dropout=lora_dropout, lora_target=lora_target,
        additional_target=additional_target, create_new_adapter=create_new_adapter
    ))

    with gr.Accordion(label="RLHF config", open=False) as rlhf_tab:
        with gr.Row():
            dpo_beta = gr.Slider(value=0.1, minimum=0, maximum=1, step=0.01, scale=1)
            reward_model = gr.Dropdown(scale=3, allow_custom_value=True)
            refresh_btn = gr.Button(scale=1)

    refresh_btn.click(
        list_adapters,
        [engine.manager.get_elem_by_name("top.model_name"), engine.manager.get_elem_by_name("top.finetuning_type")],
        [reward_model],
        queue=False
    )

    input_elems.update({dpo_beta, reward_model})
    elem_dict.update(dict(rlhf_tab=rlhf_tab, dpo_beta=dpo_beta, reward_model=reward_model, refresh_btn=refresh_btn))

    with gr.Row():
        cmd_preview_btn = gr.Button()
        save_script_btn = gr.Button()
        start_btn = gr.Button()
        stop_btn = gr.Button()

    with gr.Row():
        output_dir = gr.Textbox()
        cache_dir = gr.Textbox(value='/aigc/dataclub/cache')
        scripts_file = gr.Dropdown(choices=get_scripts(), value=0, scale=1)

    with gr.Accordion(label="损失", open=False):
        with gr.Row():
            loss_viewer = gr.Plot()

    with gr.Row():
        resume_btn = gr.Checkbox(visible=False, interactive=False, value=False)
        process_bar = gr.Slider(visible=False, interactive=False)

    with gr.Box():
        output_box = gr.Markdown()


    output_box.postprocess = output_postprocess
    input_elems.update({output_dir, cache_dir, scripts_file})
    output_elems = [output_box, process_bar]

    cmd_preview_btn.click(engine.runner.preview_train, input_elems, output_elems)
    save_script_btn.click(engine.runner.save_scripts,  input_elems, scripts_file)
    start_btn.click(engine.runner.run_train, input_elems, output_elems)
    stop_btn.click(engine.runner.set_abort, queue=False)
    resume_btn.change(engine.runner.monitor, outputs=output_elems)
    scripts_file.change(engine.runner.preview_script, input_elems, output_elems)

    elem_dict.update(dict(
        cmd_preview_btn=cmd_preview_btn, start_btn=start_btn, stop_btn=stop_btn, output_dir=output_dir,
        resume_btn=resume_btn, process_bar=process_bar, output_box=output_box, loss_viewer=loss_viewer,
        cache_dir=cache_dir,scripts_file=scripts_file,save_script_btn=save_script_btn
    ))

    output_box.change(
        gen_plot,
        [
            engine.manager.get_elem_by_name("top.model_name"),
            engine.manager.get_elem_by_name("top.finetuning_type"),
            output_dir
        ],
        loss_viewer,
        queue=False
    )

    return elem_dict
