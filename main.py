import tkinter as tk
from jipaiqi import JiPaiQi, CARD_PIONTS, CARD_COLORS, CARDS


class Table(tk.Toplevel):
    def __init__(self, x, y, rows, cols):
        super(Table, self).__init__()
        self.rows = rows
        self.cols = cols
        self.cells = []
        self.frame1 = tk.Frame(self, borderwidth=1)
        #self.frame2 = tk.Frame(self, borderwidth=1)
        for row in range(self.rows):
            self.cells.append([])
            for col in range(self.cols):
                cell = tk.Label(self.frame1, height=1, width=2, relief=tk.GROOVE, font=('Segoe UI Emoji', 13))
                cell.grid(row=row, column=col)
                self.cells[-1].append(cell)
        self.detail_info = tk.Text(self.frame1, height=1, width=30, relief=tk.GROOVE, font=('Segoe UI Emoji', 13))
        # self.lacks.configure()
        self.detail_info.tag_configure("gold", foreground="black", background="gold")
        self.detail_info.tag_configure("red", foreground="red")
        self.detail_info.tag_configure("black", foreground="black")
        self.detail_info.tag_configure("blue", foreground="blue")
        self.detail_info.config(state=tk.DISABLED)
        self.detail_info.grid(row=self.rows, column=0, columnspan=self.cols, sticky=tk.W + tk.E)
        self.frame1.pack(fill=tk.BOTH)


        # self.frame2.pack(fill=tk.BOTH)
        # self.geometry('%dx%d+%d+%d' % (302, 142, x, y))
        self.geometry('+%d+%d' % (x, y))
        self.overrideredirect(True)
        self.attributes('-alpha', 0.8)
        self.attributes('-topmost', True)


out_card_tables = [
    Table(1000, 60, 5, 15),
    Table(20, 200, 5, 15),
    Table(1000, 850, 5, 15),
    Table(1480, 200, 5, 15),
]


def update_ui(jipaiqi):
    from jipaiqi import jipaiqi_to_table_content
    content = jipaiqi_to_table_content(jipaiqi)
    if content is None:
        return
    tables, details = content
    for player in range(4):
        for row in range(5):
            for col in range(15):
                text, bg, fg = tables[player][row][col]
                if text == '⒑':
                    text = '10'
                out_card_tables[player].cells[row][col].config(
                    text=text,
                    background=bg,
                    foreground=fg
                )

    for player in range(4):
        s = details[player]
        out_card_tables[player].detail_info.config(state=tk.NORMAL)
        out_card_tables[player].detail_info.delete('1.0', tk.END)
        for c in s:
            if c == '主':
                tag = 'gold'
            elif c in '♠♣':
                tag = 'black'
            elif c in '♥♦':
                tag = 'red'
            else:
                tag = 'blue'
            out_card_tables[player].detail_info.insert(tk.END, c, tag)
        out_card_tables[player].detail_info.config(state=tk.DISABLED)


root = out_card_tables[0].master
jipaiqi = JiPaiQi(img_dir=None, do_record=True, loop=True)
# jipaiqi = JiPaiQi(img_dir='1644511845', do_record=True, loop=False)
def task():
    has_update = jipaiqi.step()
    if has_update:
        update_ui(jipaiqi)
    root.after(100, task)
root.after(100, task)
# root.attributes('-topmost', True)
# root.bind("<KeyPress>", lambda _: print(out_card_tables[0].winfo_height(), out_card_tables[0].winfo_width()))
tk.mainloop()



