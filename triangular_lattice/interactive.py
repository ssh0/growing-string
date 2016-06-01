#! /usr/bin/env python
# -*- coding:utf-8 -*-
#
# written by ssh0, October 2014.

from __future__ import print_function

__doc__ = '''Jupyter Notebook like Gtk wrapper class.

You can create Scalebar, Switch, ComboBox via simple interface.

usage example:

>>> from gtk_wrapper import interactive
>>> def f(a, b=20):
...     return a + b
...
>>> w = interactive(f, a=(0, 20), b=20)

(you can add buttons and label here manually)

(then, you should add the next line)
>>> w.display()

(and you can get the result for f(a, b) by)
32

(or get the arguments with a dictionary)
>>> w.result
>>> w.kwargs
{'a': 8, 'b': 24}
'''

from gi.repository import Gtk
import inspect


class Interactive(Gtk.Window):

    def __init__(self, func, title='title', **kwargs):
        self.__doc__ = __doc__
        self.func = func
        self.kwargs = dict()
        args, varargs, keywords, defaults = inspect.getargspec(self.func)
        d = []
        if defaults:
            for default in defaults:
                d.append(default)
            self.kwdefaults = dict(zip(args[len(args) - len(defaults):], d))
        else:
            self.kwdefaults = dict()

        Gtk.Window.__init__(self, title=title)
        hbox = Gtk.Box(spacing=6)
        self.add(hbox)

        self.listbox = Gtk.ListBox()
        self.listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        hbox.pack_start(self.listbox, True, True, 0)
        self.status = Gtk.Label()

        for kw, arg in kwargs.items():
            kw = str(kw)
            arg_type = type(arg)
            if arg_type == tuple:
                # type check for elements in tuple
                argtype = self.type_check(arg)
                if argtype == str:
                    self.combobox_str(kw, arg)
                else:
                    self.scale_bar(kw, arg, argtype)
            elif arg_type == int or arg_type == float:
                self.scale_bar(kw, [arg], arg_type)
            elif arg_type == bool:
                self.switch(kw, arg)
            elif arg_type == str:
                self.label(arg)
                self.kwargs[kw] = arg
            elif arg_type == dict:
                self.combobox_dict(kw, arg)
            else:
                raise TypeError

        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        hbox.pack_start(self.status, True, True, 10)
        self.status.set_text(str(self.kwargs))
        self.listbox.add(row)

    def display(self):
        self.connect("delete-event", Gtk.main_quit)
        self.show_all()
        Gtk.main()

    def status_change(self):
        self.status.set_text(str(self.kwargs))
        self.kwargs_for_function = self.kwdefaults
        self.kwargs_for_function.update(self.kwargs)
        self.result = self.func(**self.kwargs_for_function)

    def set_value(self, kw, new_value):
        # if argument is already given in func, use it for a default one
        if kw in self.kwdefaults:
            self.kwargs[kw] = self.kwdefaults[kw]
            return self.kwdefaults[kw]
        else:
            self.kwargs[kw] = new_value
            return new_value

    def type_check(self, arg):
        argtype = type(arg[0])
        if not all([type(a) == argtype for a in arg]):
            raise TypeError("""types in a tuple must be the same.
            int or float: Scalebar
            str         : Combobox""")
        return argtype

    def add_label(self, kw, parent):
        label = Gtk.Label(kw, xalign=0)
        parent.pack_start(label, True, True, 10)

    def scale_bar(self, kw, arg, argtype):

        def scale_interact(scale, _type):
            if _type == int:
                self.kwargs[kw] = int(scale.get_value())
            else:
                self.kwargs[kw] = float(scale.get_value())
            self.status_change()

        # length check for tuple
        len_arg = len(arg)
        if len_arg > 3 or len_arg == 0:
            raise IndexError("tuple must be 1 or 2 or 3 element(s)")

        if argtype == int:
            scale_digit = 0
        elif argtype == float:
            scale_digit = 2
        else:
            raise TypeError("arg must be int or float")

        # set the values
        if len_arg == 3:
            scale_from = arg[0]
            scale_to = arg[1]
            scale_digit = arg[2]
        elif len_arg == 2:
            scale_from = arg[0]
            scale_to = arg[1]
        else:
            scale_from = arg[0] * (-1)
            scale_to = arg[0] * 3

        # create scale widget in listbox
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)

        label = Gtk.Label(kw, xalign=0)
        hbox.pack_start(label, False, True, 10)
        scale = Gtk.Scale()
        scale.set_range(scale_from, scale_to)
        scale.set_digits(scale_digit)
        scale.set_value(self.set_value(kw, arg[0]))
        scale.set_draw_value(True)
        scale.connect('value-changed', scale_interact, argtype)
        hbox.pack_start(scale, True, True, 10)

        self.listbox.add(row)

    def switch(self, kw, arg):

        def on_switch_activated(switch, gparam):
            self.kwargs[kw] = switch.get_active()
            self.status_change()

        # create switch widget in listbox
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        self.add_label(kw, hbox)
        switch = Gtk.Switch()
        switch.connect("notify::active", on_switch_activated)
        switch.set_active(self.set_value(kw, arg))
        hbox.pack_start(switch, False, False, 10)
        self.listbox.add(row)

    def combobox_str(self, kw, arg):

        def on_combo_changed(combo):
            tree_iter = combo.get_active()
            if tree_iter is not None:
                self.kwargs[kw] = arg[tree_iter]
                self.status_change()

        argstore = Gtk.ListStore(str)
        for a in arg:
            argstore.append([a])

        # create combobox widget in listbox
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)
        self.add_label(kw, hbox)
        combo = Gtk.ComboBox.new_with_model(argstore)
        combo.connect("changed", on_combo_changed)
        renderer_text = Gtk.CellRendererText()
        combo.pack_start(renderer_text, True)
        combo.add_attribute(renderer_text, "text", 0)
        combo.set_active(arg.index(self.set_value(kw, arg)))
        hbox.pack_start(combo, False, False, True)
        self.listbox.add(row)

    def combobox_dict(self, kw, arg):

        def on_combo_changed(combo):
            tree_iter = combo.get_active()
            if tree_iter is not None:
                self.kwargs[kw] = values[tree_iter]
                self.status_change()

        argstore = Gtk.ListStore(str)
        keys = list(arg.keys())
        values = list(arg.values())
        for a in keys:
            argstore.append([a])

        # create combobox widget in listbox
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        row.add(hbox)

        self.add_label(kw, hbox)
        combo = Gtk.ComboBox.new_with_model(argstore)
        combo.connect("changed", on_combo_changed)
        renderer_text = Gtk.CellRendererText()
        combo.pack_start(renderer_text, True)
        combo.add_attribute(renderer_text, "text", 0)
        combo.set_active(values.index(self.set_value(kw, arg)))
        hbox.pack_start(combo, False, False, True)

        self.listbox.add(row)


if __name__ == '__main__':
    from gi.repository import Gtk

    def f(x=12, y=20, z='III', o=False, i=20):
        print("x: {0}, y: {1}, z: {2}, o: {3}, i: {4}".format(x, y, z, o, i))

    def b1(button):
        print(w.kwargs)

    buttons = [('b1', b1), ('exit', Gtk.main_quit)]
    w = Interactive(f, x=10, y=(1., 100.),
                    z=("ZZZ", "III", "foo", "bar"),
                    i={'0': 0, '10': 10, '20': 20},
                    o=True
                    )
    row = Gtk.ListBoxRow()
    hbox = Gtk.HBox(spacing=10)
    row.add(hbox)
    for b in buttons:
        button = Gtk.Button(b[0])
        button.connect('clicked', b[1])
        hbox.pack_start(button, True, True, 0)
    w.listbox.add(row)

    w.display()
