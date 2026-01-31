import subprocess
import platform
from pycocoevalcap.spice.spice import Spice

class SpiceScorer(Spice):
    """
    Override de Spice pour pouvoir gerer la RAM conssomée et fonctionner sur Windows
    
    """
    def compute_score(self, gts, res, java_path):
        
        # Détection de l'OS pour utiliser ou non le cache avec Spice
        current_os = platform.system()
        # Impossible sur Windows
        is_windows = current_os == 'Windows'
        
        original_check_call = subprocess.check_call

        def mocked_check_call(cmd, **kwargs):
            """
            Fonction d'override de l'environement et de la librairie Spice pour eviter de planter sur Windows ou avec peu de ram
            """
            new_cmd = []
            
            if cmd[0] == 'java':
                new_cmd.append(str(java_path))
            else:
                new_cmd.append(str(cmd[0]))

            iterator = iter(cmd[1:])
            for arg in iterator:
                if arg == '-Xmx8G':
                    # réduction de la RAM
                    new_cmd.append('-Xmx4G')
                
                # debug
                elif arg == '-silent':
                    continue 
                
                elif arg == '-cache':
                    if is_windows:
                        # si windows on n'utilise pas le cache
                        next(iterator, None) 
                        continue
                    else:
                        new_cmd.append(arg)
                
                else:
                    new_cmd.append(str(arg))
            
            new_cmd.insert(1, '-Duser.language=en')
            new_cmd.insert(2, '-Duser.country=US')
            
            try:
                result = subprocess.run(
                    new_cmd,
                    cwd=kwargs.get('cwd'),
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                return 0
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8', errors='replace')
                print(f"CRASH JAVA SPICE ({error_msg})")
                raise e

        subprocess.check_call = mocked_check_call
        
        try:
            return super().compute_score(gts, res)
        except Exception:
            return 0.0, []
        finally:
            subprocess.check_call = original_check_call