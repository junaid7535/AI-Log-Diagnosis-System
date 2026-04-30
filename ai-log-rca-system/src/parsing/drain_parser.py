import re
from typing import List, Dict, Tuple
from collections import defaultdict
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import hashlib

class DrainParser:
    def __init__(self, depth: int = 4, sim_th: float = 0.4, max_children: int = 100):
        config = TemplateMinerConfig()
        config.depth = depth
        config.sim_th = sim_th
        config.max_children = max_children
        self.template_miner = TemplateMiner(config=config)
        self.template_cache = {}
        
    def parse(self, log_line: str) -> Tuple[str, Dict]:
        """Extract log template and parameters"""
        result = self.template_miner.add_log_message(log_line)
        
        if result['change_type'] != 'none':
            template = result['cluster']['template']
            template_id = hashlib.md5(template.encode()).hexdigest()[:8]
        else:
            template = result['cluster']['template']
            template_id = result['cluster']['cluster_id']
        
        # Extract variable parameters
        params = self._extract_params(log_line, template)
        
        return template, {
            'template_id': template_id,
            'parameters': params,
            'cluster_size': result['cluster']['size']
        }
    
    def _extract_params(self, log_line: str, template: str) -> Dict:
        """Extract variable values from log line using template"""
        template_parts = template.split()
        log_parts = log_line.split()
        params = {}
        
        for i, (t_part, l_part) in enumerate(zip(template_parts, log_parts)):
            if '<*>' in t_part or t_part.startswith('<'):
                params[f'param_{i}'] = l_part
        
        return params