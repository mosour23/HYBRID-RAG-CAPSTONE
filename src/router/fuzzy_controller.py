import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyLogicController:
    def __init__(self):
        # 1. Define inputs (from QueryAnalyzer) and outputs (routing decision)
        # Semantic density from 0.0 to 1.0
        self.sd = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'semantic_density')
        
        # Multi-hop requirement from 0.0 to 1.0
        self.mhr = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'multihop_req')
        
        # Output: final complexity score from 0 to 100
        self.routing_score = ctrl.Consequent(np.arange(0, 101, 1), 'routing_score')

        # 2. Configure membership functions - triangular, as proposed
        # Semantic density: low, moderate, high
        self.sd['low'] = fuzz.trimf(self.sd.universe, [0, 0, 0.4])
        self.sd['moderate'] = fuzz.trimf(self.sd.universe, [0.2, 0.5, 0.8])
        self.sd['high'] = fuzz.trimf(self.sd.universe, [0.6, 1.0, 1.0])

        # Multi-hop requirement: low, moderate, high
        self.mhr['low'] = fuzz.trimf(self.mhr.universe, [0, 0, 0.5])
        self.mhr['moderate'] = fuzz.trimf(self.mhr.universe, [0.3, 0.5, 0.7])
        self.mhr['high'] = fuzz.trimf(self.mhr.universe, [0.5, 1.0, 1.0])

        # Output (routing decision): RAG (low score) or Long-Context (high score)
        self.routing_score['RAG'] = fuzz.trimf(self.routing_score.universe, [0, 0, 60])
        self.routing_score['LongContext'] = fuzz.trimf(self.routing_score.universe, [40, 100, 100])

        # 3. Build the fuzzy logic rule base
        rule1 = ctrl.Rule(self.sd['low'] & self.mhr['low'], self.routing_score['RAG'])
        rule2 = ctrl.Rule(self.sd['moderate'] & self.mhr['low'], self.routing_score['RAG'])
        rule3 = ctrl.Rule(self.sd['low'] & self.mhr['moderate'], self.routing_score['RAG'])
        
        # Rules that lead to the long-context route
        rule4 = ctrl.Rule(self.sd['high'] | self.mhr['high'], self.routing_score['LongContext'])
        rule5 = ctrl.Rule(self.sd['moderate'] & self.mhr['moderate'], self.routing_score['LongContext'])

        # 4. Assemble the system
        self.routing_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.router = ctrl.ControlSystemSimulation(self.routing_ctrl)

    def decide_route(self, sd_value: float, mhr_value: float) -> dict:
        """
        Receive the scores, pass them to the fuzzy logic system, and return the final decision.
        """
        # Pass the inputs to the system
        self.router.input['semantic_density'] = sd_value
        self.router.input['multihop_req'] = mhr_value

        # Run the computations
        self.router.compute()

        # Extract the result
        final_score = self.router.output['routing_score']
        
        # Make the final decision based on the threshold
        # If the score is above 50, route to long-context; otherwise, use RAG
        decision = "Long-Context" if final_score > 50 else "OP-RAG"

        return {
            "score": round(final_score, 2),
            "decision": decision
        }
        