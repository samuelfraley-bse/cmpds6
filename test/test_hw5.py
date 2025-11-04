"""
Unit tests for hw5.py
Tests for Patient, Card, Deck, Triangle, Rectangle, Circle classes
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import pytest

from hw5 import Card, Circle, Deck, Patient, PlaneFigure, Rectangle, Triangle

# run the tests in terminal with: pytest test/test_hw5.py -v
# run test coverage with:
#                          pip install pytest-cov
#                          pytest test/test_hw5.py --cov=hw5

# ========================================================== tests coverage ===========================================================
# _________________________________________ coverage: platform darwin, python 3.13.0-final-0 __________________________________________

# Name     Stmts   Miss  Cover
# ----------------------------
# hw5.py      74      2    97%
# ----------------------------
# TOTAL       74      2    97%

# ============================================================================
# EXERCISE 1: Patient Class Tests
# ============================================================================


class TestPatient:
    """Test suite for the Patient class"""

    def test_patient_creation(self):
        """Test that a patient can be created with name and symptoms"""
        patient = Patient("John Doe", ["fever", "cough"])

        assert patient.name == "John Doe"
        assert patient.symptoms == ["fever", "cough"]
        assert patient.tests == {}  # Should start with empty tests dict

    def test_patient_with_empty_symptoms(self):
        """Test creating a patient with no symptoms"""
        patient = Patient("Jane Smith", [])

        assert patient.name == "Jane Smith"
        assert patient.symptoms == []
        assert len(patient.tests) == 0

    def test_patient_with_multiple_symptoms(self):
        """Test patient with many symptoms"""
        symptoms = ["fever", "cough", "anosmia", "headache", "fatigue"]
        patient = Patient("Bob", symptoms)

        assert len(patient.symptoms) == 5
        assert "fever" in patient.symptoms
        assert "anosmia" in patient.symptoms

    def test_add_test_positive(self):
        """Test adding a positive test result"""
        patient = Patient("Alice", ["fever"])
        patient.add_test("covid", True)

        assert "covid" in patient.tests
        assert patient.tests["covid"] == True

    def test_add_test_negative(self):
        """Test adding a negative test result"""
        patient = Patient("Bob", ["headache"])
        patient.add_test("covid", False)

        assert "covid" in patient.tests
        assert patient.tests["covid"] == False

    def test_add_multiple_tests(self):
        """Test adding multiple different tests"""
        patient = Patient("Carol", ["fever"])
        patient.add_test("covid", True)
        patient.add_test("flu", False)
        patient.add_test("strep", True)

        assert len(patient.tests) == 3
        assert patient.tests["covid"] == True
        assert patient.tests["flu"] == False
        assert patient.tests["strep"] == True

    def test_add_test_overwrite(self):
        """Test that adding same test twice overwrites the first result"""
        patient = Patient("Dave", ["cough"])
        patient.add_test("covid", False)
        patient.add_test("covid", True)  # Overwrite

        assert patient.tests["covid"] == True
        assert len(patient.tests) == 1

    def test_has_covid_with_positive_test(self):
        """Test has_covid returns 0.99 for positive COVID test"""
        patient = Patient("Eve", ["headache"])
        patient.add_test("covid", True)

        assert patient.has_covid() == 0.99

    def test_has_covid_with_negative_test(self):
        """Test has_covid returns 0.01 for negative COVID test"""
        patient = Patient("Frank", ["headache"])
        patient.add_test("covid", False)

        assert patient.has_covid() == 0.01

    def test_has_covid_no_symptoms_no_test(self):
        """Test base probability of 0.05 with no COVID symptoms"""
        patient = Patient("Grace", ["headache", "nausea"])

        probability = patient.has_covid()
        assert probability == pytest.approx(0.05, abs=1e-10)

    def test_has_covid_one_symptom(self):
        """Test probability increases by 0.1 for one COVID symptom"""
        patient = Patient("Henry", ["fever"])

        probability = patient.has_covid()
        assert probability == pytest.approx(0.15, abs=1e-10)  # 0.05 + 0.1

    def test_has_covid_two_symptoms(self):
        """Test probability with two COVID symptoms"""
        patient = Patient("Iris", ["fever", "cough"])

        probability = patient.has_covid()
        assert probability == pytest.approx(0.25, abs=1e-10)  # 0.05 + 0.2

    def test_has_covid_all_three_symptoms(self):
        """Test probability with all three COVID symptoms"""
        patient = Patient("Jack", ["fever", "cough", "anosmia"])

        probability = patient.has_covid()
        assert probability == pytest.approx(0.35, abs=1e-10)  # 0.05 + 0.3

    def test_has_covid_symptoms_plus_other_symptoms(self):
        """Test that only COVID symptoms count"""
        patient = Patient("Karen", ["fever", "cough", "headache", "nausea"])

        probability = patient.has_covid()
        assert probability == pytest.approx(0.25, abs=1e-10)  # Only fever + cough count

    def test_has_covid_test_overrides_symptoms(self):
        """Test that test result takes priority over symptoms"""
        patient = Patient("Larry", ["fever", "cough", "anosmia"])
        # Without test, would be 0.35
        patient.add_test("covid", False)

        assert patient.has_covid() == 0.01  # Test result takes priority


# ============================================================================
# EXERCISE 2: Card and Deck Classes Tests
# ============================================================================


class TestCard:
    """Test suite for the Card class"""

    def test_card_creation(self):
        """Test that a card can be created with suit and value"""
        card = Card("Hearts", "A")

        assert card.suit == "Hearts"
        assert card.value == "A"

    def test_card_str_method(self):
        """Test the __str__ method returns correct format"""
        card = Card("Spades", "K")

        assert str(card) == "K of Spades"

    def test_card_different_suits(self):
        """Test cards with different suits"""
        hearts = Card("Hearts", "10")
        diamonds = Card("Diamonds", "J")
        clubs = Card("Clubs", "Q")
        spades = Card("Spades", "K")

        assert hearts.suit == "Hearts"
        assert diamonds.suit == "Diamonds"
        assert clubs.suit == "Clubs"
        assert spades.suit == "Spades"

    def test_card_different_values(self):
        """Test cards with different values"""
        ace = Card("Hearts", "A")
        two = Card("Hearts", "2")
        ten = Card("Hearts", "10")
        jack = Card("Hearts", "J")

        assert ace.value == "A"
        assert two.value == "2"
        assert ten.value == "10"
        assert jack.value == "J"


class TestDeck:
    """Test suite for the Deck class"""

    def test_deck_creation(self):
        """Test that deck is created with 52 cards"""
        deck = Deck()

        assert len(deck.cards) == 52

    def test_deck_has_all_suits(self):
        """Test that deck contains all four suits"""
        deck = Deck()
        suits = {card.suit for card in deck.cards}

        assert suits == {"Hearts", "Diamonds", "Clubs", "Spades"}

    def test_deck_has_all_values(self):
        """Test that deck contains all 13 values"""
        deck = Deck()
        values = {card.value for card in deck.cards}

        expected_values = {
            "A",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "J",
            "Q",
            "K",
        }
        assert values == expected_values

    def test_deck_has_correct_card_count_per_suit(self):
        """Test that each suit has exactly 13 cards"""
        deck = Deck()

        for suit in ["Hearts", "Diamonds", "Clubs", "Spades"]:
            suit_cards = [card for card in deck.cards if card.suit == suit]
            assert len(suit_cards) == 13

    def test_deck_has_correct_card_count_per_value(self):
        """Test that each value appears exactly 4 times"""
        deck = Deck()

        for value in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
            value_cards = [card for card in deck.cards if card.value == value]
            assert len(value_cards) == 4

    def test_shuffle_maintains_card_count(self):
        """Test that shuffle doesn't lose or duplicate cards"""
        deck = Deck()
        deck.shuffle()

        assert len(deck.cards) == 52

    def test_draw_returns_card(self):
        """Test that draw returns a Card object"""
        deck = Deck()
        card = deck.draw()

        assert isinstance(card, Card)

    def test_draw_removes_card_from_deck(self):
        """Test that drawing a card removes it from the deck"""
        deck = Deck()
        initial_count = len(deck.cards)

        deck.draw()

        assert len(deck.cards) == initial_count - 1

    def test_draw_multiple_cards(self):
        """Test drawing multiple cards"""
        deck = Deck()

        for i in range(10):
            card = deck.draw()
            assert card is not None
            assert len(deck.cards) == 52 - i - 1

    def test_draw_all_cards(self):
        """Test drawing all 52 cards"""
        deck = Deck()
        drawn_cards = []

        for _ in range(52):
            card = deck.draw()
            drawn_cards.append(card)

        assert len(drawn_cards) == 52
        assert len(deck.cards) == 0

    def test_draw_from_empty_deck(self):
        """Test that drawing from empty deck returns None"""
        deck = Deck()

        # Draw all cards
        for _ in range(52):
            deck.draw()

        # Try to draw from empty deck
        card = deck.draw()
        assert card is None


# ============================================================================
# EXERCISE 3: PlaneFigure Classes Tests
# ============================================================================


class TestPlaneFigure:
    """Test suite for the abstract PlaneFigure class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that PlaneFigure cannot be instantiated directly"""
        with pytest.raises(TypeError):
            fig = PlaneFigure()


class TestTriangle:
    """Test suite for the Triangle class"""

    def test_triangle_creation(self):
        """Test creating a triangle with valid parameters"""
        triangle = Triangle(base=5, c1=3, c2=4, h=2.4)

        assert triangle.base == 5
        assert triangle.c1 == 3
        assert triangle.c2 == 4
        assert triangle.h == 2.4

    def test_triangle_perimeter_simple(self):
        """Test triangle perimeter calculation"""
        triangle = Triangle(base=3, c1=4, c2=5, h=2)

        perimeter = triangle.compute_perimeter()
        assert perimeter == 12  # 3 + 4 + 5

    def test_triangle_surface_simple(self):
        """Test triangle area calculation"""
        triangle = Triangle(base=6, c1=4, c2=5, h=3)

        area = triangle.compute_surface()
        assert area == pytest.approx(9.0)  # (6 * 3) / 2

    def test_triangle_is_instance_of_planefigure(self):
        """Test that Triangle is an instance of PlaneFigure"""
        triangle = Triangle(base=3, c1=4, c2=5, h=2)

        assert isinstance(triangle, PlaneFigure)


class TestRectangle:
    """Test suite for the Rectangle class"""

    def test_rectangle_creation(self):
        """Test creating a rectangle with valid parameters"""
        rectangle = Rectangle(a=5, b=10)

        assert rectangle.a == 5
        assert rectangle.b == 10

    def test_rectangle_perimeter_simple(self):
        """Test rectangle perimeter calculation"""
        rectangle = Rectangle(a=5, b=10)

        perimeter = rectangle.compute_perimeter()
        assert perimeter == 30  # 2 * (5 + 10)

    def test_rectangle_surface_simple(self):
        """Test rectangle area calculation"""
        rectangle = Rectangle(a=5, b=10)

        area = rectangle.compute_surface()
        assert area == 50  # 5 * 10

    def test_rectangle_is_instance_of_planefigure(self):
        """Test that Rectangle is an instance of PlaneFigure"""
        rectangle = Rectangle(a=4, b=6)

        assert isinstance(rectangle, PlaneFigure)


class TestCircle:
    """Test suite for the Circle class"""

    def test_circle_creation(self):
        """Test creating a circle with valid radius"""
        circle = Circle(radius=5)

        assert circle.radius == 5

    def test_circle_perimeter_simple(self):
        """Test circle circumference calculation"""
        circle = Circle(radius=5)

        perimeter = circle.compute_perimeter()
        expected = 2 * math.pi * 5

        assert perimeter == pytest.approx(expected)

    def test_circle_surface_simple(self):
        """Test circle area calculation"""
        circle = Circle(radius=5)

        area = circle.compute_surface()
        expected = math.pi * 25

        assert area == pytest.approx(expected)

    def test_circle_is_instance_of_planefigure(self):
        """Test that Circle is an instance of PlaneFigure"""
        circle = Circle(radius=3)

        assert isinstance(circle, PlaneFigure)


# ============================================================================
# POLYMORPHISM TESTS
# ============================================================================


class TestPolymorphism:
    """Test that all shapes can be used polymorphically"""

    def test_all_shapes_have_compute_perimeter(self):
        """Test that all shapes implement compute_perimeter"""
        shapes = [Triangle(3, 4, 5, 2), Rectangle(4, 6), Circle(5)]

        for shape in shapes:
            perimeter = shape.compute_perimeter()
            assert isinstance(perimeter, (int, float))
            assert perimeter > 0

    def test_all_shapes_have_compute_surface(self):
        """Test that all shapes implement compute_surface"""
        shapes = [Triangle(3, 4, 5, 2), Rectangle(4, 6), Circle(5)]

        for shape in shapes:
            area = shape.compute_surface()
            assert isinstance(area, (int, float))
            assert area > 0
