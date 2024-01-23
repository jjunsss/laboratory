## **Using the Main Function with Distributed Data Parallel (DDP) utils**

The ddp_utils are designed to be quick and easy to use when you want to split your data thorugh multiple GPUs. Plz refers to the [[`target_examples.py`](https://github.com/jjunsss/laboratory/blob/main/DDP/target_examples.py)]

### **Key Components**

- **`ddp_utils.ddp_data_split()`**: This utility function is used to split the dataset evenly across the available GPUs.
- **`@ddp_utils.ddp_on_and_off`**: A decorator used to initialize and terminate the DDP environment before and after the execution of the main function, ensuring a clean setup and shutdown.
- `ddp_utils` :  The `get_world_size` / `get_rank` / g`et_local_size` / `get_local_rank`, `is_main_process,`etc. that you need while using DDP are available directly from ddp_utils.

### **Notes**

- The provided code is a basic template. Depending on your specific requirements, you might need to modify the image processing logic within the for loop.
- Please see ddp_utils for detailed usage and modification according to your requirements.
- Please use the `target_examples.py` first.
- Also, you can refer to my blog contents for a more detailed understanding of the basic DDP descriptions. [[LINK](https://blog.naver.com/jjunsss/222920508815); Ready to 5 steps]
