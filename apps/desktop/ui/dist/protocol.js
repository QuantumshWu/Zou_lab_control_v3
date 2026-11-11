/** The kinds a board offers, in the order the plotting owner gives them (the live board's and the Figure Viewer's). */
export const boardKinds = (kinds) => kinds.filter(kind => typeof kind.board === "number").sort((left, right) => left.board - right.board);
