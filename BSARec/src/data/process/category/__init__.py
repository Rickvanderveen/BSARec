class DatasetCategoryMapCreator:
    @classmethod
    def create(cls, category_type: str):
        if category_type == "Native":
            cls.create_native_map()
        elif category_type == "Popularity":
            print("Creating item-popularity map")
            cls.create_popularity_map()
        else:
            raise NotImplementedError
    
    @staticmethod
    def create_native_map():
        raise NotImplementedError

    @staticmethod
    def create_popularity_map():
        raise NotImplementedError
