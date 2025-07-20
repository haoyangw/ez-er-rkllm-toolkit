from rkllm.api import RKLLM
from huggingface_hub import login, whoami, snapshot_download, auth_check, ModelCard, HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from pathlib import Path
import inquirer
import shutil
import os

class RKLLMRemotePipeline:
    def __init__(self, model_id="", lora_id="", platform="rk3588",
                 qtype="w8a8", quantized_algorithm="normal", hybrid_rate="0.0", library_type="HF",
                 optimization=1, max_context=""):
        """
        Initialize primary values for pipeline class.

        :param model_id: HuggingFace repository ID for model (required)
        :param lora_id: Same as model_id, but for LoRA (optional)
        :param platform: CPU type of target platform. Must be rk3588 or rk3576
        :param optimization: 1 means "optimize model" and 0 means "don't optimize" - may incrase performance,
            at the expense of accuracy
        :param qtype: either a string or list of quantization types
        :param quantized_algorithm: Quantization accuracy optimization algorithm,
            options are 'normal', 'grq' and 'gdq'. 'normal' supports all quantization types,
            while 'grq' and 'gdq' only support 'w4a16' and grouped 'w4a16' quantization
        :param hybrid_rate: block(group-wise quantization) ratio, whose value is between 
            0 and 1, 0 indicating the disable of mixed quantization
        :param max_context: Maximum context length for converted model, up to 16,384 and
            must be divisible by 32
        """
        self.model_id = model_id
        self.lora_id = lora_id
        self.platform = platform
        self.qtype = qtype
        self.quantized_algorithm = quantized_algorithm
        self.hybrid_rate = hybrid_rate
        self.library_type = library_type
        self.optimization = optimization
        self.max_context = max_context

    @staticmethod
    def mkpath(path):
        """
        HuggingFace Hub will just fail if the local_dir you are downloading to does not exist
        RKLLM will also fail to export if the directory does not already exist.

        :param paths: a list of paths (as strings) to check and create
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                print(f"mkdir'd {path}")
            else:
                print(f"Path {path} already exists! Great job!")
        except RuntimeError as e:
            print(f"Can't create paths for importing and exporting model.\n{e}")

    @staticmethod
    def cleanup_models(path=Path("./models")):
        print(f"Cleaning up model directory...")
        shutil.rmtree(path)

    def user_inputs(self):
        '''
        Obtain necessary inputs for model generation
        This remote pipeline downloads the selected model and LoRA (if one is selected.)
        It then iterates through and exports a model for each quantization type.
        '''
        self.inputs = [
            inquirer.Text("model_id", 
                          message="HuggingFace Repo ID for Model in user/repo format (default is Qwen/Qwen2.5-7B-Instruct)", 
                          default="Qwen/Qwen2.5-7B-Instruct"),
            inquirer.Text("lora_id", 
                          message="HuggingFace LoRA ID for Model in user/repo format",
                          default=None),
            inquirer.List("library", 
                          message="HuggingFace or GGUF Format?", 
                          choices=["HF","GGUF"], 
                          default="HF"),
            inquirer.List("platform", 
                          message="Which platform would you like to build for?", 
                          choices=["rk3588", "rk3576"], 
                          default="rk3588"),
            inquirer.List("optimization",
                          message="Optimize model?\nBetter performance, longer conversion\n0=None\n1=Yes",
                          choices=[0, 1]),
            inquirer.List("qtype", 
                              message="Quantization type?", 
                              choices=["w8a8", "w8a8_g128", "w8a8_g256", "w8a8_g512"], 
                              ignore=lambda x: not x["platform"] == "rk3588"),
            inquirer.List("qtype", 
                              message="Quantization type?", 
                              choices=["w8a8", "w4a16", "w4a16_g32", "w4a16_g64", "w4a16_g128"], 
                              ignore=lambda x: not x["platform"] == "rk3576"),
            inquirer.Text("hybrid_rate",
                          message="Block (group-wise quantization) ratio, whose value is between 0 and 1, 0 indicating none",
                          default="0.0"),
            inquirer.Text("max_context",
                          message="Maximum context length for converted model, up to 16,384 and must be divisible by 32",
                          default="")
        ]
        
        self.config = inquirer.prompt(self.inputs)
        
        self.model_id = self.config["model_id"]
        self.lora_id = self.config["lora_id"]
        self.platform = self.config["platform"]
        self.optimization = int(self.config["optimization"])
        self.qtype = self.config["qtype"]
        self.hybrid_rate = float(self.config["hybrid_rate"])
        self.library_type = self.config["library"]
        self.max_context = self.config["max_context"]
        
    def build_vars(self):
        if self.platform == "rk3588":
            self.npu_cores = 3
        elif self.platform == "rk3576":
            self.npu_cores = 2
        self.dataset = None
        self.qparams = None
        self.device = "cpu"
        self.model_name = self.model_id.split("/", 1)[1]
        self.model_dir = f"./models/{self.model_name}/"
        self.name_suffix = f"{self.platform}-{self.qtype}-opt-{self.optimization}-hybrid-ratio-{self.hybrid_rate}"
        if self.lora_id == "":
            self.lora_name = None
            self.lora_dir = None
            self.lorapath = None
            self.export_name = f"{self.model_name}-{self.name_suffix}"
            self.export_path = f"./models/{self.model_name}-{self.platform}/"
        else:
            self.lora_name = self.lora_id.split("/", 1)[1]
            self.lora_dir = f"./models/{self.lora_name}/"
            self.export_name = f"{self.model_name}-{self.lora_name}-{self.name_suffix}"
            self.export_path = f"./models/{self.model_name}-{self.lora_name}-{self.platform}/"
        self.rkllm_version = "1.2.1"

    def remote_pipeline_to_local(self):
        '''
        Full conversion pipeline
        Downloads the chosen model from HuggingFace to a local destination, so no need
        to copy from the local HF cache.
        '''
        print(f"Checking if {self.model_dir} and {self.export_path} exist...")
        self.mkpath(self.model_dir)
        self.mkpath(self.export_path)

        print(f"Loading base model {self.model_id} from HuggingFace and downloading to {self.model_dir}")
        self.modelpath = snapshot_download(repo_id=self.model_id, local_dir=self.model_dir)

        if self.lora_id == None:
            print(f"LoRA is {self.lora_id} - skipping download")
        else:
            print (f"Downloading LoRA: {self.lora_id} from HuggingFace to {self.lora_dir}")
            try:
                self.lorapath = snapshot_download(repo_id=self.lora_id, local_dir=self.lora_dir)
            except:
                print(f"Downloading LoRA failed. Omitting from export.")
                self.lorapath == None

        print("Initializing RKLLM class...")
        self.rkllm = RKLLM()
        
        if self.library_type == "HF":
            print(f"Have to load model for each config")
            status = self.rkllm.load_huggingface(model=self.modelpath, model_lora=self.lorapath, 
                                                device=self.device)
            if status != 0:
                raise RuntimeError(f"Failed to load model: {status}")
            else:
                print(f"{self.model_name} loaded successfully!")    
        elif self.library_type == "GGUF":
            print(f"Have to load model for each config")
            status = self.rkllm.load_gguf(model=self.modelpath)
            if status != 0:
                raise RuntimeError(f"Failed to load model: {status}")
            else:
                print(f"{self.model_name} loaded successfully!")
        else:
            print("Model must be of type HF (HuggingFace) or GGUF.")
            raise RuntimeError("Must be something wrong with the selector! Try again!")

        print(f"Building {self.model_name} with {self.qtype} quantization and optmization level {self.optimization}")
        if self.max_context:
        	self.max_context = int(self.max_context)
        	status = self.rkllm.build(do_quantization=True, optimization_level=self.optimization,
        							quantized_dtype=self.qtype, quantized_algorithm=self.quantized_algorithm,
        							hybrid_rate=self.hybrid_rate, target_platform=self.platform,
        							num_npu_core=self.npu_cores, extra_qparams=self.qparams,
        							dataset=self.dataset, max_context=self.max_context)
        else:
        	status = self.rkllm.build(do_quantization=True, optimization_level=self.optimization,
        							quantized_dtype=self.qtype, quantized_algorithm=self.quantized_algorithm,
        							hybrid_rate=self.hybrid_rate, target_platform=self.platform,
        							num_npu_core=self.npu_cores, extra_qparams=self.qparams,
        							dataset=self.dataset)
        if status != 0:
            raise RuntimeError(f"Failed to build model: {status}")
        else:
            print(f"{self.model_name} built successfully!")
    
        status = self.rkllm.export_rkllm(f"{self.export_path}{self.export_name}.rkllm")
        if status != 0:
            raise RuntimeError(f"Failed to export model: {status}")
        else:
            print(f"{self.model_name} exported successfully to {self.export_path}!")


# Don't trust super().__init__ here    
class HubHelpers:
    def __init__(self, platform, model_id, lora_id, qtype, rkllm_version):
        """
        Collection of helpers for interacting with HuggingFace.
        Due to some weird memory leak-y behaviors observed, would rather pass down
        parameters from the pipeline class then try to do something with super().__init__

        :param platform: CPU type of target platform. Must be rk3588 or rk3576
        :param model_id: HuggingFace repository ID for model (required)
        :param lora_id: Same as model_id, but for LoRA (optional)
        :param rkllm_version: version of RKLLM used for conversion.
        """
        self.model_id = model_id
        self.lora_id = lora_id
        self.platform = platform
        self.qtype = qtype
        self.models = {"base": model_id, "lora": lora_id}
        self.rkllm_version = rkllm_version
        self.home_dir = os.environ['HOME']
        # Use Rust implementation of transfer for moar speed
        os.environ['HF_HUB_ENABLE_HF_TRANSFER']='1'

    @staticmethod
    def repo_check(model):
        """
        Checks if a HuggingFace repo exists and is gated
        """
        try:
            auth_check(model)
        except GatedRepoError:
            # Handle gated repository error
            print(f"{model} is a gated repo.\nYou do not have permission to access it.\n \
                  Please authenticate.\n")
        except RepositoryNotFoundError:
            # Handle repository not found error
            print(f"{model} not found.")
        else:
            print(f"Model repo {model} has been validated!")
            return True   
    
    def login_to_hf(self):
        """
        Helper function to authenticate with HuggingFace.
        Necessary for downloading gated repositories, and uploading.
        """
        self.token_path = f"{self.home_dir}/.cache/huggingface/token"
        if os.path.exists(self.token_path):
            self.token_file = open(self.token_path, "r")
            self.hf_token = self.token_file.read()
        else:
            self.hf_input = [inquirer.Text("token", message="Please enter your Hugging Face token", default="")]
            self.hf_token = inquirer.prompt(self.hf_input)["token"]
        try:
            login(token=self.hf_token)
        except Exception as e:
            print(f"Login failed: {e}\nGated models will be inaccessible, and you " + \
                  "will not be able to upload to HuggingFace.")
        else:
            print("Logged into HuggingFace!\n")
            self.hf_username = whoami(self.hf_token)["name"]
            print(self.hf_username + "\n")
            
    def build_card(self, export_path):
        """
        Inserts text into the README.md file of the original model, after the model data. 
        Using the HF built-in functions kept omitting the card's model data,
        so gonna do this old school.
        """
        self.model_name = self.model_id.split("/", 1)[1]
        self.card_in = ModelCard.load(self.model_id)
        self.card_out = export_path + "README.md"
        self.template = f'---\n' + \
            f'{self.card_in.data.to_yaml()}\n' + \
            f'---\n' + \
            f'# {self.model_name}-{self.platform.upper()}-{self.rkllm_version}\n\n' + \
            f'This version of {self.model_name} has been converted to run on the {self.platform.upper()} NPU using {self.qtype} quantization.\n\n' + \
            f'This model has been optimized with the following LoRA: {self.lora_id}\n\n' + \
            f'Compatible with RKLLM version: {self.rkllm_version}\n\n' + \
            f'## Useful links:\n' + \
            f'[Official RKLLM GitHub](https://github.com/airockchip/rknn-llm) \n\n' + \
            f'[RockhipNPU Reddit](https://reddit.com/r/RockchipNPU) \n\n' + \
            f'[EZRKNN-LLM](https://github.com/Pelochus/ezrknn-llm/) \n\n' + \
            f'Pretty much anything by these folks: [marty1885](https://github.com/marty1885) and [happyme531](https://huggingface.co/happyme531) \n\n' + \
            f'Converted using https://github.com/c0zaut/ez-er-rkllm-toolkit \n\n' + \
            f'# Original Model Card for base model, {self.model_name}, below:\n\n' + \
            f'{self.card_in.text}'
        try:
            ModelCard.save(self.template, self.card_out)
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
        except RuntimeWarning as w:
            print(f"Runtime Warning: {w}")
        else:
            print(f"Model card successfully exported to {self.card_out}!")
            c = open(self.card_out, 'r')
            print(c.read())
            c.close()

    def upload_to_repo(self, model, import_path, export_path):
        self.hf_api = HfApi(token=self.hf_token)
        self.repo_id = f"{self.hf_username}/{model}-{self.platform}-{self.rkllm_version}"
        
        print(f"Creating repo if it does not already exist")
        try:
            self.repo_url = self.hf_api.create_repo(exist_ok=True, repo_id=self.repo_id)
        except:
            print(f"Create repo for {model} failed!")
        else:
            print(f"Repo created! URL: {self.repo_url}")

        print(f"Generating model card and copying configs")
        self.build_card(export_path)
        self.import_path = Path(import_path)
        self.export_path = Path(export_path)
        print(f"Uploading files to repo")
        shutil.copytree(self.import_path, self.export_path, ignore=shutil.ignore_patterns('*.safetensors', '*.pt*', '*.bin', '*.gguf', 'README*'), 
                        copy_function=shutil.copy2, dirs_exist_ok=True)
        self.commit_info = self.hf_api.upload_folder(repo_id=self.repo_id, folder_path=self.export_path)
        print(self.commit_info)

if __name__ == "__main__":

    rk = RKLLMRemotePipeline()
    rk.user_inputs()
    rk.build_vars()
    hf = HubHelpers(platform=rk.platform, model_id=rk.model_id, lora_id=rk.lora_id, 
                    qtype=rk.qtype, rkllm_version=rk.rkllm_version)
    hf.login_to_hf()
    hf.repo_check(rk.model_id)

    try:
        rk.remote_pipeline_to_local()
    except RuntimeError as e:
        print(f"Model conversion failed: {e}")
    

    hf.upload_to_repo(model=rk.model_name, import_path=rk.model_dir, export_path=rk.export_path)
    print("Okay, these models are really big!")
    rk.cleanup_models("./models")
