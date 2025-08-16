(this doc is incomplete, using as a checklist/outline in development)

New tests for 1.6.0:
    - Gen tests:
        x Lora refiner
        x depth map through cnet node
        x style image through hints
    - LoRA node:
        - Versioning
            x Values are loaded correctly from previous version worklow
            x Widget values are saved by key
            x Widget values are loaded by key
            - On loading old version, inputs are fixed
        - UI tests
            x "Show mode" toggles visibility of "mode" widgets
            x "More" adds extra lora slots to list
                - their values are reset to defualt
                - maxes at 8
                - button is disabled at 8
                - works correctly with show mode on and off
            x "Less" removes lora slots from list
                - their values are reset (when serialised to json)
                - doesn't remove the first lora
                - button is disabled at 1
    - ControlNet node
        - Versioning
            x Values are loaded correctly from previous version worklow
            x Widget values are saved by key
            x Widget values are loaded by key
        - Ui tests
            - Correct widgets are shown depending on cnet model
    - Image Hints node
        - doesn't need ui tests, gen tests cover
