import os
import subprocess
import threading

class MeteorScorer:
    def __init__(self, java_path="java"):
        import pycocoevalcap.meteor.meteor as meteor_script_module
        
        base_path = os.path.dirname(os.path.abspath(meteor_script_module.__file__))
        self.meteor_jar = os.path.join(base_path, 'meteor-1.5.jar')
        
        if not os.path.exists(self.meteor_jar):
             potential = os.path.join(base_path, 'data', 'meteor-1.5.jar')
             if os.path.exists(potential):
                 self.meteor_jar = potential
             else:
                 raise FileNotFoundError(f"Impossible de trouver 'meteor-1.5.jar'.")

        if java_path is None: java_path = "java"

        self.meteor_cmd = [
            java_path, '-Duser.language=en', '-Duser.country=US', 
            '-jar', '-Xmx2G', self.meteor_jar, 
            '-', '-', '-stdio', '-l', 'en', '-norm'
        ]

        self.process = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(self.meteor_jar),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=os.environ 
        )
        self.lock = threading.Lock()

    def _clean_input(self, text):
        """Nettoyage agressif pour éviter la désynchronisation"""
        if not isinstance(text, str): return str(text)
        # Supprime sauts de ligne et séparateurs METEOR
        return text.replace('\n', ' ').replace('\r', ' ').replace('|||', '').strip()

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = self._clean_input(hypothesis_str)
        reference_list = [self._clean_input(ref) for ref in reference_list]
        
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        
        self.process.stdin.write(f"{score_line}\n".encode('utf-8'))
        self.process.stdin.flush()
        return self.process.stdout.readline().decode('utf-8').strip()

    def compute_score(self, gts, res):
        imgIds = sorted(list(gts.keys()))
        eval_line = 'EVAL'
        
        with self.lock:
            for i in imgIds:
                try:
                    stat = self._stat(res[i][0], gts[i])
                    eval_line += ' ||| {}'.format(stat)
                except Exception:
                    # Si une image plante, on met une stat vide pour ne pas casser le reste
                    eval_line += ' ||| 0.0'

            self.process.stdin.write(f"{eval_line}\n".encode('utf-8'))
            self.process.stdin.flush()
            
            score_str = self.process.stdout.readline().decode('utf-8').strip()
            
            # Gestion des réponses parasites (le bug du '14.0 9.0...')
            if " " in score_str: 
                # On essaie de lire la ligne suivante, parfois le score est caché derrière
                try:
                    next_line = self.process.stdout.readline().decode('utf-8').strip()
                    if next_line and " " not in next_line: score_str = next_line
                except: pass

            try:
                return float(score_str.replace(',', '.')), []
            except:
                return 0.0, []

    def close(self):
        if self.process:
            try:
                self.process.kill()
                self.process.wait()
            except: pass
    
    def __del__(self):
        self.close()