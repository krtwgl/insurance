from sklearn.base import BaseEstimator, TransformerMixin

class InsuranceFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tier_1_cities = {
            "Mumbai", "Delhi", "Bangalore", "Chennai",
            "Kolkata", "Hyderabad", "Pune"
        }
        self.tier_2_cities = {
            "Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi",
            "Visakhapatnam", "Coimbatore", "Bhopal", "Nagpur", "Vadodara",
            "Surat", "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi",
            "Agra", "Dehradun", "Mysore", "Jabalpur", "Guwahati",
            "Thiruvananthapuram", "Ludhiana", "Nashik", "Allahabad",
            "Udaipur", "Aurangabad", "Hubli", "Belgaum", "Salem",
            "Vijayawada", "Tiruchirappalli", "Bhavnagar", "Gwalior",
            "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
            "Warangal", "Kolhapur", "Bilaspur", "Jalandhar", "Noida",
            "Guntur", "Asansol", "Siliguri"
        }

    def _get_age_group(self, age):
        if age < 25:
            return "young"
        elif age < 45:
            return "adult"
        elif age < 60:
            return "middle_aged"
        return "senior"

    def _get_lifestyle_risk(self, smoker, bmi):
        if smoker and bmi > 30:
            return "high"
        elif smoker or bmi > 27:
            return "medium"
        return "low"

    def _get_city_tier(self, city):
        if city in self.tier_1_cities:
            return "1"
        elif city in self.tier_2_cities:
            return "2"
        return "3"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Vectorized calculations
        df["bmi"] = df["weight"] / (df["height"] ** 2)
        df["age_group"] = df["age"].apply(self._get_age_group)
        df["lifestyle_risk"] = [
            self._get_lifestyle_risk(s, b) for s, b in zip(df["smoker"], df["bmi"])
        ]
        df["city_tier"] = df["city"].apply(self._get_city_tier)

        selected_features = [
            "bmi", "age_group", "lifestyle_risk",
            "city_tier", "income_lpa", "occupation"
        ]
        return df[selected_features]