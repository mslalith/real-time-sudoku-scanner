#
# This is written by svinec (taken from stackoverflow)
#

import random
import math

# mySudoku.solve("A", "0", time.time(), False)
class Sudoku:
    def __init__(self, _g=None):
        if _g is None:
            _g = []
        self._input_grid = []  # store a copy of the original input grid for later use
        self.grid = []  # this is the main grid that will be iterated
        for i in _g:  # copy the nested lists by value, otherwise Python keeps the reference for the nested lists
            self._input_grid.append(i[:])
            self.grid.append(i[:])

        self.empty_cells = set()  # set of all currently empty cells (by index number from left to right, top to bottom)
        self.empty_cells_initial = set()  # this will be used to compare against the current set of empty cells in
        # order to determine if all cells have been iterated
        self.current_cell = None  # used for iterating
        self.current_choice = 0  # used for iterating
        self.history = []  # list of visited cells for backtracking
        self.choices = {}  # dictionary of sets of currently available digits for each cell
        self.nextCellWeights = {}  # a dictionary that contains weights for all cells, used when making a choice of next cell
        self.nextCellWeights_1 = lambda x: None  # the first function that will be called to assign weights
        self.nextCellWeights_2 = lambda x: None  # the second function that will be called to assign weights
        self.nextChoiceWeights = {}  # a dictionary that contains weights for all choices, used when selecting the next choice
        self.nextChoiceWeights_1 = lambda x: None  # the first function that will be called to assign weights
        self.nextChoiceWeights_2 = lambda x: None  # the second function that will be called to assign weights

        self.search_space = 1  # the number of possible combinations among the empty cells only, for information purpose only
        self.iterations = 0  # number of main iterations, for information purpose only
        self.iterations_backtrack = 0  # number of backtrack iterations, for information purpose only

        self.digit_heuristic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
                                9: 0}  # store the number of times each digit is used in order to choose the ones that are least/most used, parameter "3" and "4"
        self.centerWeights = {}  # a dictionary of the distances for each cell from the center of the grid, calculated only once at the beginning

        # populate centerWeights by using Pythagorean theorem
        for id in range(81):
            row = id // 9
            col = id % 9
            self.centerWeights[id] = int(round(100 * math.sqrt((row - 4) ** 2 + (col - 4) ** 2)))

    # for debugging purposes
    def dump(self, _custom_text, _file_object):
        _custom_text += ", cell: {}, choice: {}, choices: {}, empty: {}, history: {}, grid: {}\n".format(
            self.current_cell, self.current_choice, self.choices, self.empty_cells, self.history, self.grid)
        _file_object.write(_custom_text)

    # to be called before each solve of the grid
    def reset(self):
        self.grid = []
        for i in self._input_grid:
            self.grid.append(i[:])

        self.empty_cells = set()
        self.empty_cells_initial = set()
        self.current_cell = None
        self.current_choice = 0
        self.history = []
        self.choices = {}

        self.nextCellWeights = {}
        self.nextCellWeights_1 = lambda x: None
        self.nextCellWeights_2 = lambda x: None
        self.nextChoiceWeights = {}
        self.nextChoiceWeights_1 = lambda x: None
        self.nextChoiceWeights_2 = lambda x: None

        self.search_space = 1
        self.iterations = 0
        self.iterations_backtrack = 0

        self.digit_heuristic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def validate(self):
        # validate all rows
        for x in range(9):
            digit_count = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            for y in range(9):
                digit_count[self.grid[x][y]] += 1
            for i in digit_count:
                if digit_count[i] != 1:
                    return False

        # validate all columns
        for x in range(9):
            digit_count = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            for y in range(9):
                digit_count[self.grid[y][x]] += 1
            for i in digit_count:
                if digit_count[i] != 1:
                    return False

        # validate all 3x3 quadrants
        def validate_quadrant(_grid, from_row, to_row, from_col, to_col):
            digit_count = {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
            for x in range(from_row, to_row + 1):
                for y in range(from_col, to_col + 1):
                    digit_count[_grid[x][y]] += 1
            for i in digit_count:
                if digit_count[i] != 1:
                    return False
            return True

        for x in range(0, 7, 3):
            for y in range(0, 7, 3):
                if not validate_quadrant(self.grid, x, x + 2, y, y + 2):
                    return False
        return True

    def setCell(self, _id, _value):
        row = _id // 9
        col = _id % 9
        self.grid[row][col] = _value

    def getCell(self, _id):
        row = _id // 9
        col = _id % 9
        return self.grid[row][col]

    # returns a set of IDs of all blank cells that are related to the given one, related means from the same row, column or quadrant
    def getRelatedBlankCells(self, _id):
        result = set()
        row = _id // 9
        col = _id % 9

        for i in range(9):
            if self.grid[row][i] == 0: result.add(row * 9 + i)
        for i in range(9):
            if self.grid[i][col] == 0: result.add(i * 9 + col)
        for x in range((row // 3) * 3, (row // 3) * 3 + 3):
            for y in range((col // 3) * 3, (col // 3) * 3 + 3):
                if self.grid[x][y] == 0: result.add(x * 9 + y)

        return set(result)  # return by value

    # get the next cell to iterate
    def getNextCell(self):
        self.nextCellWeights = {}
        for id in self.empty_cells:
            self.nextCellWeights[id] = 0

        self.nextCellWeights_1(
            1000)  # these two functions will always be called, but behind them will be a different weight function depending on the optimization parameters provided
        self.nextCellWeights_2(1)

        return min(self.nextCellWeights, key=self.nextCellWeights.get)

    def nextCellWeights_A(self, _factor):  # the first cell from left to right, from top to bottom
        for id in self.nextCellWeights:
            self.nextCellWeights[id] += id * _factor

    def nextCellWeights_B(self, _factor):  # the first cell from right to left, from bottom to top
        self.nextCellWeights_A(_factor * -1)

    def nextCellWeights_C(self, _factor):  # a randomly chosen cell
        for id in self.nextCellWeights:
            self.nextCellWeights[id] += random.randint(0, 999) * _factor

    def nextCellWeights_D(self, _factor):  # the closest cell to the center of the grid
        for id in self.nextCellWeights:
            self.nextCellWeights[id] += self.centerWeights[id] * _factor

    def nextCellWeights_E(self, _factor):  # the cell that currently has the fewest choices available
        for id in self.nextCellWeights:
            self.nextCellWeights[id] += len(self.getChoices(id)) * _factor

    def nextCellWeights_F(self, _factor):  # the cell that currently has the most choices available
        self.nextCellWeights_E(_factor * -1)

    def nextCellWeights_G(self, _factor):  # the cell that has the fewest blank related cells
        for id in self.nextCellWeights:
            self.nextCellWeights[id] += len(self.getRelatedBlankCells(id)) * _factor

    def nextCellWeights_H(self, _factor):  # the cell that has the most blank related cells
        self.nextCellWeights_G(_factor * -1)

    def nextCellWeights_I(self, _factor):  # the cell that is closest to all filled cells
        for id in self.nextCellWeights:
            weight = 0
            for check in range(81):
                if self.getCell(check) != 0:
                    weight += math.sqrt((id // 9 - check // 9) ** 2 + (id % 9 - check % 9) ** 2)

    def nextCellWeights_J(self, _factor):  # the cell that is furthest from all filled cells
        self.nextCellWeights_I(_factor * -1)

    def nextCellWeights_K(self, _factor):  # the cell whose related blank cells have the fewest available choices
        for id in self.nextCellWeights:
            weight = 0
            for id_blank in self.getRelatedBlankCells(id):
                weight += len(self.getChoices(id_blank))
            self.nextCellWeights[id] += weight * _factor

    def nextCellWeights_L(self, _factor):  # the cell whose related blank cells have the most available choices
        self.nextCellWeights_K(_factor * -1)

    # for a given cell return a set of possible digits within the Sudoku restrictions
    def getChoices(self, _id):
        available_choices = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        row = _id // 9
        col = _id % 9

        # exclude digits from the same row
        for y in range(0, 9):
            if self.grid[row][y] in available_choices:
                available_choices.remove(self.grid[row][y])

        # exclude digits from the same column
        for x in range(0, 9):
            if self.grid[x][col] in available_choices:
                available_choices.remove(self.grid[x][col])

        # exclude digits from the same quadrant
        for x in range((row // 3) * 3, (row // 3) * 3 + 3):
            for y in range((col // 3) * 3, (col // 3) * 3 + 3):
                if self.grid[x][y] in available_choices:
                    available_choices.remove(self.grid[x][y])

        if len(available_choices) == 0:
            return set()
        else:
            return set(available_choices)  # return by value

    def nextChoice(self):
        self.nextChoiceWeights = {}
        for i in self.choices[self.current_cell]:
            self.nextChoiceWeights[i] = 0

        self.nextChoiceWeights_1(1000)
        self.nextChoiceWeights_2(1)

        self.current_choice = min(self.nextChoiceWeights, key=self.nextChoiceWeights.get)
        self.setCell(self.current_cell, self.current_choice)
        self.choices[self.current_cell].remove(self.current_choice)

    def nextChoiceWeights_0(self, _factor):  # the lowest digit
        for i in self.nextChoiceWeights:
            self.nextChoiceWeights[i] += i * _factor

    def nextChoiceWeights_1(self, _factor):  # the highest digit
        self.nextChoiceWeights_0(_factor * -1)

    def nextChoiceWeights_2(self, _factor):  # a randomly chosen digit
        for i in self.nextChoiceWeights:
            self.nextChoiceWeights[i] += random.randint(0, 999) * _factor

    def nextChoiceWeights_3(self, _factor):  # heuristically, the least used digit across the board
        self.digit_heuristic = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for id in range(81):
            if self.getCell(id) != 0: self.digit_heuristic[self.getCell(id)] += 1
        for i in self.nextChoiceWeights:
            self.nextChoiceWeights[i] += self.digit_heuristic[i] * _factor

    def nextChoiceWeights_4(self, _factor):  # heuristically, the most used digit across the board
        self.nextChoiceWeights_3(_factor * -1)

    def nextChoiceWeights_5(self,
                            _factor):  # the digit that will cause related blank cells to have the least number of choices available
        cell_choices = {}
        for id in self.getRelatedBlankCells(self.current_cell):
            cell_choices[id] = self.getChoices(id)

        for c in self.nextChoiceWeights:
            weight = 0
            for id in cell_choices:
                weight += len(cell_choices[id])
                if c in cell_choices[id]: weight -= 1
            self.nextChoiceWeights[c] += weight * _factor

    def nextChoiceWeights_6(self,
                            _factor):  # the digit that will cause related blank cells to have the most number of choices available
        self.nextChoiceWeights_5(_factor * -1)

    def nextChoiceWeights_7(self,
                            _factor):  # the digit that is the least common available choice among related blank cells
        cell_choices = {}
        for id in self.getRelatedBlankCells(self.current_cell):
            cell_choices[id] = self.getChoices(id)

        for c in self.nextChoiceWeights:
            weight = 0
            for id in cell_choices:
                if c in cell_choices[id]: weight += 1
            self.nextChoiceWeights[c] += weight * _factor

    def nextChoiceWeights_8(self,
                            _factor):  # the digit that is the most common available choice among related blank cells
        self.nextChoiceWeights_7(_factor * -1)

    def nextChoiceWeights_9(self, _factor):  # the digit that is the least common available choice across the board
        cell_choices = {}
        for id in range(81):
            if self.getCell(id) == 0:
                cell_choices[id] = self.getChoices(id)

        for c in self.nextChoiceWeights:
            weight = 0
            for id in cell_choices:
                if c in cell_choices[id]: weight += 1
            self.nextChoiceWeights[c] += weight * _factor

    def nextChoiceWeights_a(self, _factor):  # the digit that is the most common available choice across the board
        self.nextChoiceWeights_9(_factor * -1)

    # the main function to be called
    def solve(self, _nextCellMethod, _nextChoiceMethod, _start_time, _prefillSingleChoiceCells=False):
        s = self
        s.reset()

        # initialize optimization functions based on the optimization parameters provided
        """
        A - the first cell from left to right, from top to bottom
        B - the first cell from right to left, from bottom to top
        C - a randomly chosen cell
        D - the closest cell to the center of the grid
        E - the cell that currently has the fewest choices available
        F - the cell that currently has the most choices available
        G - the cell that has the fewest blank related cells
        H - the cell that has the most blank related cells
        I - the cell that is closest to all filled cells
        J - the cell that is furthest from all filled cells
        K - the cell whose related blank cells have the fewest available choices
        L - the cell whose related blank cells have the most available choices
        """
        if _nextCellMethod[0] in "ABCDEFGHIJKLMN":
            s.nextCellWeights_1 = getattr(s, "nextCellWeights_" + _nextCellMethod[0])
        elif _nextCellMethod[0] == " ":
            s.nextCellWeights_1 = lambda x: None
        else:
            print("(A) Incorrect optimization parameters provided")
            return False

        if len(_nextCellMethod) > 1:
            if _nextCellMethod[1] in "ABCDEFGHIJKLMN":
                s.nextCellWeights_2 = getattr(s, "nextCellWeights_" + _nextCellMethod[1])
            elif _nextCellMethod[1] == " ":
                s.nextCellWeights_2 = lambda x: None
            else:
                print("(B) Incorrect optimization parameters provided")
                return False
        else:
            s.nextCellWeights_2 = lambda x: None

        # initialize optimization functions based on the optimization parameters provided
        """
        0 - the lowest digit
        1 - the highest digit
        2 - a randomly chosen digit
        3 - heuristically, the least used digit across the board
        4 - heuristically, the most used digit across the board
        5 - the digit that will cause related blank cells to have the least number of choices available
        6 - the digit that will cause related blank cells to have the most number of choices available
        7 - the digit that is the least common available choice among related blank cells
        8 - the digit that is the most common available choice among related blank cells
        9 - the digit that is the least common available choice across the board
        a - the digit that is the most common available choice across the board
        """
        if _nextChoiceMethod[0] in "0123456789a":
            s.nextChoiceWeights_1 = getattr(s, "nextChoiceWeights_" + _nextChoiceMethod[0])
        elif _nextChoiceMethod[0] == " ":
            s.nextChoiceWeights_1 = lambda x: None
        else:
            print("(C) Incorrect optimization parameters provided")
            return False

        if len(_nextChoiceMethod) > 1:
            if _nextChoiceMethod[1] in "0123456789a":
                s.nextChoiceWeights_2 = getattr(s, "nextChoiceWeights_" + _nextChoiceMethod[1])
            elif _nextChoiceMethod[1] == " ":
                s.nextChoiceWeights_2 = lambda x: None
            else:
                print("(D) Incorrect optimization parameters provided")
                return False
        else:
            s.nextChoiceWeights_2 = lambda x: None

        # fill in all cells that have single choices only, and keep doing it until there are no left, because as soon as one cell is filled this might bring the choices down to 1 for another cell
        if _prefillSingleChoiceCells == True:
            while True:
                next = False
                for id in range(81):
                    if s.getCell(id) == 0:
                        cell_choices = s.getChoices(id)
                        if len(cell_choices) == 1:
                            c = cell_choices.pop()
                            s.setCell(id, c)
                            next = True
                if not next: break

        # initialize set of empty cells
        for x in range(0, 9, 1):
            for y in range(0, 9, 1):
                if s.grid[x][y] == 0:
                    s.empty_cells.add(9 * x + y)
        s.empty_cells_initial = set(s.empty_cells)  # copy by value

        # calculate search space
        for id in s.empty_cells:
            s.search_space *= len(s.getChoices(id))

        # initialize the iteration by choosing a first cell
        if len(s.empty_cells) < 1:
            if s.validate():
                print("Sudoku provided is valid!")
                return True
            else:
                print("Sudoku provided is not valid!")
                return False
        else:
            s.current_cell = s.getNextCell()

        s.choices[s.current_cell] = s.getChoices(s.current_cell)
        if len(s.choices[s.current_cell]) < 1:
            print("(C) Sudoku cannot be solved!")
            return False

        # start iterating the grid
        while True:
            # if time.time() - _start_time > 2.5: return False # used when doing mass tests and don't want to wait hours for an inefficient optimization to complete

            s.iterations += 1

            # if all empty cells and all possible digits have been exhausted, then the Sudoku cannot be solved
            if s.empty_cells == s.empty_cells_initial and len(s.choices[s.current_cell]) < 1:
                print("(A) Sudoku cannot be solved!")
                return False

            # if there are no empty cells, it's time to validate the Sudoku
            if len(s.empty_cells) < 1:
                if s.validate():
                    # print("Sudoku has been solved! ")
                    # print("search space is {}".format(self.search_space))
                    # print("empty cells: {}, iterations: {}, backtrack iterations: {}".format(
                    #     len(self.empty_cells_initial), self.iterations, self.iterations_backtrack))
                    # for i in range(9):
                    #     print(self.grid[i])
                    return True

            # if there are empty cells, then move to the next one
            if len(s.empty_cells) > 0:

                s.current_cell = s.getNextCell()  # get the next cell
                s.history.append(s.current_cell)  # add the cell to history
                s.empty_cells.remove(s.current_cell)  # remove the cell from the empty queue
                s.choices[s.current_cell] = s.getChoices(s.current_cell)  # get possible choices for the chosen cell

                if len(s.choices[
                           s.current_cell]) > 0:  # if there is at least one available digit, then choose it and move to the next iteration, otherwise the iteration continues below with a backtrack
                    s.nextChoice()
                    continue

            # if all empty cells have been iterated or there are no empty cells, and there are still some remaining choices, then try another choice
            if len(s.choices[s.current_cell]) > 0 and (
                    s.empty_cells == s.empty_cells_initial or len(s.empty_cells) < 1):
                s.nextChoice()
                continue

            # if none of the above, then we need to backtrack to a cell that was previously iterated
            # first, restore the current cell...
            s.history.remove(s.current_cell)  # ...by removing it from history
            s.empty_cells.add(s.current_cell)  # ...adding back to the empty queue
            del s.choices[s.current_cell]  # ...scrapping all choices
            s.current_choice = 0
            s.setCell(s.current_cell, s.current_choice)  # ...and blanking out the cell

            # ...and then, backtrack to a previous cell
            while True:
                s.iterations_backtrack += 1

                if len(s.history) < 1:
                    print("(B) Sudoku cannot be solved!")
                    return False

                s.current_cell = s.history[
                    -1]  # after getting the previous cell, do not recalculate all possible choices because we will lose the information about has been tried so far

                if len(s.choices[
                           s.current_cell]) < 1:  # backtrack until a cell is found that still has at least one unexplored choice...
                    s.history.remove(s.current_cell)
                    s.empty_cells.add(s.current_cell)
                    s.current_choice = 0
                    del s.choices[s.current_cell]
                    s.setCell(s.current_cell, s.current_choice)
                    continue

                # ...and when such cell is found, iterate it
                s.nextChoice()
                break  # and break out from the backtrack iteration but will return to the main iteration
