MOCK_CONFIG_YAML_ELABORATE = """
path_templates:
  weights_blob:
    template: "{project_name}/models/{model_name}/weights/{weights_file}"
    uses: weights_file
  temp_weights:
    template: "/local_disk0/tmp/{model_name}/weights/{weights_file}"
    uses: weights_file
  config_path:
    template: "{project_name}/models/{model_name}/config/{config_file}"
    uses: config_file
  # This template uses a variable we will deliberately not provide
  unresolved_template:
    template: "/data/{unresolved_var}/{model_name}.dat"
    uses: weights_file

experiment_name: mock_exp
experiment_version: 1
batch_size: 64

models:
  yolo:
    template_vars:
      weights_file:
        best: "yolo_best.pt"
        large: "yolo_large.pt"

  snn:
    vars:
      model_name: "snn_override" # Test overriding the model name
    template_vars:
      weights_file:
        best: "snn_best.pt"
      config_file:
        default: "snn_config.yml"

  # This model has no template variables and should be loaded as-is
  simple_model:
    some_value: 123
    some_setting: "abc"
    """
