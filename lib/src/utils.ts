export const clip = (v: number) => (v > 1 ? 1 : v < -1 ? -1 : v);

export const shuffleArray = <T = unknown>(array: Array<T>): Array<T> => {
  let currentIndex = array.length;
  let randomIndex;

  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex],
      array[currentIndex],
    ];
  }

  return array;
};
