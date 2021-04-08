###**Execute Tests:**

Run All:

`python -m pytest --disable-pytest-warnings`

Run Individual:

`python -m pytest --disable-pytest-warnings test_data_generalized.py`

`python -m pytest --disable-pytest-warnings test_data_us.py`

`python -m pytest --disable-pytest-warnings test_generative.py`

----
###**Covid Tracking Test Suite:**


**Class: TestValidationBase** 

``
General base class that contains validation functions 
for raw and procssed dataframes.
``

**Class: TestRawData**

``
Sub-class to fetch raw data and extend out for specfic raw data validation
``


**Class: TestProcessedData**


``
Sub-class to fetch raw data and extend out for specfic processed data validation.
Contains model/summary validation validation as well.
``

**Class: TestCovidTrackingBase**

``
Purpose of this class to abstract away the complexities of pulling processed/raw data
in the core test files. This makes it seamless for a the user to test either raw or processed data.
``

Example:

A user can seamlessly generate raw/processed data and validate if it's a dataframe
in the same function
````
@pytest.mark.parametrize('source_arg', ["raw", "processed"])
def test_dataframes(self, source_type):

    source_type.generate_data()
    source_type.is_dataframe()
````