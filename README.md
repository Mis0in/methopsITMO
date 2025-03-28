<h2>План выполнения лабы по метоптам</h2>

<h3> Справочная информация </h3>

1. Выбранный язык - **python**, понадобятся библиотеки <code>numpy</code>, <code>scipy</code>, <code>matplotlib</code>, <code>prettytable</code>
2. Про теорию можно почитать [ВШЭ](http://www.machinelearning.ru/wiki/images/6/6b/MO17_practice1.pdf) или [Статью](https://habr.com/ru/articles/561128/)
3. Есть ещё методы одномерного поиска - например тернарный и золотого сечения, про них можно почитать в конспекте 2 лекции

<h3> Задачи связанные с кодом </h3>

1. <s>Запарсить алгоритм градиентного спуска и написать его </s> <span style="color : blue">[М]</span>  
2. <s>Написать разные способы выбора шага (scheduling)</s> <span style="color : blue">[М]</span>  
3. <s>Написать вычисление производной</s> <span style="color : blue">[М]</span>  
4. <s>Написать алгоритм армихо</s> <span style="color : blue">[М]</span>  
5. <s>В вычисление производной в данный момент <code>dx</code> считается как вектор в котором на всех позициях нули, а на <code>i</code>-той позиции стоит <code>eps</code>. Надо поменять логику чтобы <code>dx</code> = sqrt(e) * x, где e - машинный эпсилон *(sys.float_info.epsilon)* <span style="color : blue">[М]</span> или <span style="color : orange">[Да]</span>  </s>
6. <s> Найти готовый градиентный спуск из <code>scipy.optimize</code> и добавить его в код <span style="color : blue">[М]</span> </s> 
7. <s> Добавить вывод скорости работы программы *(скоростью будем считать количество вычислений градиента)* подумать, можно ли увеличить скорость, при это не написав 500 строк кода <span style="color : orange">[Да]</span> </s> 
8. <s> Написать ещё один алгоритм одномерного поиска *(предположительно Вольфа)* <span style="color : blue">[М]</span> или <span style="color : orange">[Да]</span> </s>
9. <s> добавить рисовалку графиков   <span style="color : red">[Ди]</span>  </s>
10. <s> модифицировать градиентный спуск так, чтобы он рисовал траекторию своего обхода на графике <span style="color : red">[Ди]</span> </s>

<h3> Задачи связанные с отчётом </h3>

1. **Начать вести отчёт** - создать файл, придумать *описание постановки задачи лабораторной работы* <span style="color : green">[Н]</span>
2. **Описать используемые методы**: <q>В
описании метода укажите, это реализованный Вами или библиотечный метод,
из какой библиотеки он взят, или какую библиотеку использует внутри
реализации, если использует. При наличии, указывайте важные особенности
Вашей реализации метода, они могут быть алгоритмические (например,
имеющиеся hyperпараметры), так и технические (например, использование
мемоизации в алгоритме).</q> <span style="color : green">[Н]</span>
3. Выбрать несколько унимодальных функции двух переменных, на которых эффективность различных методов отличается. <span style="color : green">[Н]</span>
4. HARD Оформить каждую из выбранных функций: оформление должно содержать *саму функцию, начальную точку, стратегию выбора шага, какие константы были взяты, какой критерий остановки использовался,* а также результаты самого алгоритма и *сколько вычислений градиента* это заняло.
**нужно взять несколько разных параметров на одну функцию и сравнить при каких параметрах алгоритм эффективнее** <span style="color : orange">[Дa]</span>
5. Для каждой функции добавить графики, содержащие траекторию работы алгоритма, точку остановки (данный пункт можно выполнить только после соответствующей реализации в коде). Добавить сравнение с <code>scipy.optimize</code> там где это возможно <span style="color : green">[Н]</span>
6. Оформить одну любую мультимодальную функцию  <span style="color : red">[Ди]</span>
7. Оформить одну любую функцию с зашумленными значениями *(может быть получена добавлением случайной величины к функции, по идее)* <span style="color : red">[Ди]</span>
8. Сделать выводы <span style="color : green">[Н]</span>
