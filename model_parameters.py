# Model Parameters and Functions
class UnifiedHIVParameters:
    def __init__(self):
        self.spatial_params = {
            'DT': 0.1,  # T cell diffusion coefficient
            'DV': 0.2,  # Viral diffusion coefficient
            'DC': 0.15  # Cytokine diffusion coefficient
        }
        
        self.tissue_specific = {
            'blood': {
                'infection_rate': 1.0,
                'clearance_rate': 23.0,
                'immune_response': 1.0
            },
            'lymph_nodes': {
                'infection_rate': 1.5,
                'clearance_rate': 15.0,
                'immune_response': 1.2
            },
            'brain': {
                'infection_rate': 0.3,
                'clearance_rate': 5.0,
                'immune_response': 0.4
            },
            'gut': {
                'infection_rate': 2.0,
                'clearance_rate': 20.0,
                'immune_response': 1.5
            }
        }
        
        self.molecular_rates = {
            'reverse_transcription': 0.8,
            'integration': 0.6,
            'transcription': 1.2,
            'translation': 1.0,
            'assembly': 0.9,
            'budding': 0.7
        }